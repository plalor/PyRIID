import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel
from time import perf_counter as timer


class DANN(PyRIIDModel):
    """Domain Adversarial Neural Network classifier."""
    def __init__(self, activation="relu", d_optimizer=None, f_optimizer=None,
                 source_model=None, grl_hidden_layers=None, use_da_scheduler=True,
                 da_param=10, dropout=0, metrics=None):
        """
        Args:
            activation: activation function to use for discriminator dense layer
            d_optimizer: tensorflow optimizer or optimizer name for the discriminator
            f_optimizer: tensorflow optimizer or optimizer name for the feature extractor
            source_model: pretrained source model
            grl_hidden_layers: sizes of the gradient reversal dense layers
            use_da_scheduler: whether to use a scheduler for ramping up lambda
            da_param: value for gamma (if using a da scheduler), otherwise value for lambda
            dropout: dropout rate to apply to the adapted model and discriminator layers
            metrics: dict of metric functions {name: function}
        """
        super().__init__()

        self.activation = activation
        self.d_optimizer = d_optimizer or Adam(learning_rate=0.001)
        self.f_optimizer = f_optimizer or Adam(learning_rate=0.001)
        self.grl_hidden_layers = grl_hidden_layers
        self.use_da_scheduler = use_da_scheduler
        if self.use_da_scheduler:
            self.gamma = da_param
        else:
            self.lmbda = da_param
        self.discriminator_loss = BinaryCrossentropy()
        self.dropout = dropout
        self.metrics = metrics or {}

        if source_model is not None:
            self.classification_loss = source_model.loss

            def modify_dropout(layer):
                if isinstance(layer, Dropout):
                    return Dropout(self.dropout, name=layer.name)
                return layer.__class__.from_config(layer.get_config())
            
            self.source_model = clone_model(
                source_model,
                clone_function=modify_dropout
            )
            self.source_model.build(source_model.input_shape)
            self.source_model.set_weights(source_model.get_weights())
            self.source_model.compile(
                optimizer=source_model.optimizer,
                loss=source_model.loss,
                metrics=source_model.metrics
            )

            all_layers = self.source_model.layers
            feature_extractor_input = self.source_model.input
            feature_extractor_output = all_layers[-2].output
            self.feature_extractor = Model(inputs=feature_extractor_input, outputs=feature_extractor_output, name="feature_extractor")

            classifier_input = Input(shape=feature_extractor_output.shape[1:], name="feature_extractor_output")
            classifier_output = all_layers[-1](classifier_input)
            self.classifier = Model(inputs=classifier_input, outputs=classifier_output, name="classifier")
        else:
            print("WARNING: no pretrained source model was provided")

    def fit(self, source_ss: SampleSet, target_ss: SampleSet, source_val_ss: SampleSet, target_val_ss: SampleSet,
            batch_size=64, epochs=None, patience=None, es_mode="min", es_monitor="tgt_val_loss", target_level="Isotope", verbose=False, training_time=None):
        """Fit a model to the given `SampleSet`(s).

        Args:
            source_ss: `SampleSet` of `n` training spectra from the source domain where `n` >= 1
                and the spectra are either foreground (AKA, "net") or gross.
            target_ss: `SampleSet` of `n` training spectra from the target domain where `n` >= 1
                and the spectra are either foreground (AKA, "net") or gross.
            source_val_ss: `SampleSet` of `m` validation spectra from the source domain where 
                `m` >= 1 and the spectra are either foreground (AKA, "net") or gross.
            target_val_ss: `SampleSet` of `m` validation spectra from the target domain where 
                `m` >= 1 and the spectra are either foreground (AKA, "net") or gross.
            batch_size: number of samples per gradient update
            epochs: maximum number of training iterations
            patience: number of epochs to wait before early stopping
            target_level: `SampleSet.sources` column level to use
            verbose: whether to show detailed model training output
            training_time: whether to terminate early if run exceeds prealloted time

        Returns:
            `history` dictionary

        Raises:
            `ValueError` when no spectra are provided as input
        """

        if source_ss.n_samples <= 0 or target_ss.n_samples <= 0:
            raise ValueError("Empty spectr[a|um] provided!")

        if source_ss.spectra_type == SpectraType.Gross:
            self.model_inputs = (ModelInput.GrossSpectrum,)
        elif source_ss.spectra_type == SpectraType.Foreground:
            self.model_inputs = (ModelInput.ForegroundSpectrum,)
        elif source_ss.spectra_type == SpectraType.Background:
            self.model_inputs = (ModelInput.BackgroundSpectrum,)
        else:
            raise ValueError(f"{source_ss.spectra_type} is not supported in this model.")

        if training_time is None:
            training_time = np.inf
            epochs = epochs or 20

        # Preparing training and validation data
        X_source = source_ss.get_samples().astype("float32")
        X_target = target_ss.get_samples().astype("float32")

        source_contributions_df = source_ss.sources.T.groupby(target_level, sort=False).sum().T
        model_outputs = source_contributions_df.columns.values.tolist()
        Y_source = source_contributions_df.values.astype("float32")

        n_val = min(len(source_val_ss), len(target_val_ss))
        source_val_ss = source_val_ss[:n_val]
        target_val_ss = target_val_ss[:n_val]
        
        X_src_val = source_val_ss.get_samples().astype("float32")
        X_tgt_val = target_val_ss.get_samples().astype("float32")

        Y_src_val = source_val_ss.sources.T.groupby(target_level, sort=False).sum().T.values.astype("float32")
        Y_tgt_val = target_val_ss.sources.T.groupby(target_level, sort=False).sum().T.values.astype("float32")

        # Create datasets
        half_batch_size = batch_size // 2
        steps_per_epoch = min(
            len(X_source) // half_batch_size,
            len(X_target) // half_batch_size
        )
        
        source_dataset = (
            tf.data.Dataset
              .from_tensor_slices((X_source, Y_source))
              .repeat()
              .shuffle(len(X_source))
              .batch(half_batch_size)
        )
        
        target_dataset = (
            tf.data.Dataset
              .from_tensor_slices((X_target,))
              .repeat()
              .shuffle(len(X_target))
              .batch(half_batch_size)
        )
        
        dataset = (
            tf.data.Dataset
              .zip((source_dataset, target_dataset))
              .prefetch(tf.data.AUTOTUNE)
        )

        # Make validation dataset
        batch_size_val = 64
        half_batch_size_val = batch_size_val // 2
        
        src_val_dataset = (
            tf.data.Dataset
              .from_tensor_slices((X_src_val, Y_src_val))
              .batch(half_batch_size_val)
        )
        
        tgt_val_dataset = (
            tf.data.Dataset
              .from_tensor_slices((X_tgt_val, Y_tgt_val))
              .batch(half_batch_size_val)
        )
        
        val_dataset = (
            tf.data.Dataset
              .zip((src_val_dataset, tgt_val_dataset))
              .prefetch(tf.data.AUTOTUNE)
        )

        # Build discriminator
        input_shape = self.feature_extractor.output_shape[1]
        inputs = Input(shape=(input_shape,), name="features")
        self.grl_layer = GradientReversalLayer()
        grl = self.grl_layer(inputs)
        for i, nodes in enumerate(self.grl_hidden_layers):
            grl = Dense(nodes, activation=self.activation, name=f"dense_{i}")(grl)
            if self.dropout > 0:
                grl = Dropout(self.dropout, name=f"dropout_{i}")(grl)
        output = Dense(1, activation="sigmoid", name="discriminator")(grl)
        self.discriminator = Model(inputs, output, name="Discriminator")

        # Define DANN model
        self.model = Model(
            inputs=self.feature_extractor.input,
            outputs=self.classifier(self.feature_extractor.output)
        )
        self.model.compile(loss = self.classification_loss)

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs = source_ss.sources.T.groupby(target_level, sort=False).sum().T.columns.values.tolist(),
            normalization=source_ss.spectra_state,
        )

        # Training loop
        self.history = {"total_loss": [], "class_loss": [], "domain_loss": [], "src_val_loss": [], "tgt_val_loss": [], "domain_val_loss": []}
        for metric in self.metrics:
            metric_name = getattr(metric, '__name__', str(metric))
            self.history[f"src_val_{metric_name}"] = []
            self.history[f"tgt_val_{metric_name}"] = []
        
        best_metric = np.inf if es_mode == "min" else -np.inf
        best_weights = None
        wait = 0
        epoch = 0
        t0 = timer()

        it = iter(dataset)
        while True:
            epoch += 1
            if epochs is not None and epoch > epochs:
                break
            
            if verbose:
                t1 = timer()
                if epochs:
                    print(f"Epoch {epoch}/{epochs}...", end="")
                else:
                    print(f"Epoch {epoch}...", end="")

            if self.use_da_scheduler:
                if epochs is not None:
                    p = epoch / epochs
                else:
                    p = epoch / 10
                new_lmbda = 2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0
            else:
                new_lmbda = self.lmbda
            tf.keras.backend.set_value(self.grl_layer.lmbda, new_lmbda)

            total_loss_avg = tf.keras.metrics.Mean()
            class_loss_avg = tf.keras.metrics.Mean()
            domain_loss_avg = tf.keras.metrics.Mean()
            for step in range(steps_per_epoch):
                (x_s, y_s), x_t = next(it)
                domain_loss = self.train_discriminator_step(x_s, x_t)
                total_loss, class_loss = self.train_feature_extractor_step(x_s, y_s, x_t)
                total_loss_avg.update_state(total_loss)
                class_loss_avg.update_state(class_loss)
                domain_loss_avg.update_state(domain_loss)

            src_class_loss_avg = tf.keras.metrics.Mean()
            tgt_class_loss_avg = tf.keras.metrics.Mean()
            domain_val_loss_avg = tf.keras.metrics.Mean()
            
            src_metric_avgs = {name: tf.keras.metrics.Mean() for name in self.metrics}
            tgt_metric_avgs = {name: tf.keras.metrics.Mean() for name in self.metrics}
            
            for (x_s_val, y_s_val), (x_t_val, y_t_val) in val_dataset:
                y_s_pred = self.model(x_s_val, training=False)
                loss_s = self.classification_loss(y_s_val, y_s_pred)
                src_class_loss_avg.update_state(loss_s)
            
                y_t_pred = self.model(x_t_val, training=False)
                loss_t = self.classification_loss(y_t_val, y_t_pred)
                tgt_class_loss_avg.update_state(loss_t)
                
                f_s_val = self.feature_extractor(x_s_val, training=False)
                f_t_val = self.feature_extractor(x_t_val, training=False)
            
                d_s_val = self.discriminator(f_s_val, training=False)
                d_t_val = self.discriminator(f_t_val, training=False)
            
                y_s_domain = tf.zeros_like(d_s_val)
                y_t_domain = tf.ones_like(d_t_val)
            
                loss_s_domain = self.discriminator_loss(y_s_domain, d_s_val)
                loss_t_domain = self.discriminator_loss(y_t_domain, d_t_val)
                domain_val_loss = loss_s_domain + loss_t_domain
                domain_val_loss_avg.update_state(domain_val_loss)
                
                for metric_name, metric_fn in self.metrics.items():
                    src_metric = metric_fn(y_s_val.numpy(), y_s_pred.numpy())
                    tgt_metric = metric_fn(y_t_val.numpy(), y_t_pred.numpy())
                    src_metric_avgs[metric_name].update_state(src_metric)
                    tgt_metric_avgs[metric_name].update_state(tgt_metric)

            total_loss = total_loss_avg.result().numpy()
            class_loss = class_loss_avg.result().numpy()
            domain_loss = domain_loss_avg.result().numpy()
            src_val_loss = src_class_loss_avg.result().numpy()
            tgt_val_loss = tgt_class_loss_avg.result().numpy()
            domain_val_loss = domain_val_loss_avg.result().numpy()
            
            self.history["total_loss"].append(total_loss)
            self.history["class_loss"].append(class_loss)
            self.history["domain_loss"].append(domain_loss)
            self.history["src_val_loss"].append(src_val_loss)
            self.history["tgt_val_loss"].append(tgt_val_loss)
            self.history["domain_val_loss"].append(domain_val_loss)
            
            for metric in self.metrics:
                metric_name = getattr(metric, '__name__', str(metric))
                self.history[f"src_val_{metric_name}"].append(src_metric_avgs[metric].result().numpy())
                self.history[f"tgt_val_{metric_name}"].append(tgt_metric_avgs[metric].result().numpy())

            if verbose:
                print(f"Finished in {timer()-t1:.0f} seconds")
                print("  "
                      f"total_loss: {total_loss:.3g} - "
                      f"class_loss: {class_loss:.3g} - "
                      f"domain_loss: {domain_loss:.3g} - "
                      f"src_val_loss: {src_val_loss:.3g} - "
                      f"tgt_val_loss: {tgt_val_loss:.3g} - "
                      f"domain_val_loss: {domain_val_loss:.3g}")

            current_metric = self.history[es_monitor][-1]
            is_better = current_metric < best_metric if es_mode == "min" else current_metric > best_metric
            
            if is_better:
                best_metric = current_metric
                best_weights = self.model.get_weights()
                wait = 0
            else:
                wait += 1
                if patience is not None and wait > patience:
                    if verbose:
                        print(f"No improvement for {patience} epochs, stopping early.")
                    break

            if timer() - t0 > training_time:
                if verbose:
                    print("Reached preallotted training time, terminating.")
                break

        if best_weights is not None:
            self.model.set_weights(best_weights)

        return self.history

    def predict(self, ss: SampleSet, bg_ss: SampleSet = None, batch_size: int = 1000):
        """Classify the spectra in the provided `SampleSet`(s).

        Results are stored inside the first SampleSet's prediction-related properties.

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are either
                foreground (AKA, "net") or gross
            bg_ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are background
            batch_size: batch size during call to self.model.predict
        """
        x_test = ss.get_samples().astype(float)
        if bg_ss:
            X = [x_test, bg_ss.get_samples().astype(float)]
        else:
            X = x_test

        results = self.model.predict(X, batch_size=batch_size)

        col_level_idx = SampleSet.SOURCES_MULTI_INDEX_NAMES.index(self.target_level)
        col_level_subset = SampleSet.SOURCES_MULTI_INDEX_NAMES[:col_level_idx+1]
        ss.prediction_probas = pd.DataFrame(
            data=results,
            columns=pd.MultiIndex.from_tuples(
                self.get_model_outputs_as_label_tuples(),
                names=col_level_subset
            )
        )

        ss.classified_by = self.model_id

    @tf.function
    def train_discriminator_step(self, x_s, x_t):
        with tf.GradientTape() as tape:
            # extract features
            f_s = self.feature_extractor(x_s, training=False)
            f_t = self.feature_extractor(x_t, training=False)

            # run through discriminator branch
            d_s = self.discriminator(f_s, training=True)
            d_t = self.discriminator(f_t, training=True)

            # source/target domain labels
            y_s = tf.zeros_like(d_s)
            y_t = tf.ones_like(d_t)

            # binary‐crossentropy on both source and target
            loss_s = self.discriminator_loss(y_s, d_s)
            loss_t = self.discriminator_loss(y_t, d_t)
            domain_loss = loss_s + loss_t

        # gradients only on the discriminator's weights
        grads = tape.gradient(domain_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return domain_loss

    @tf.function
    def train_feature_extractor_step(self, x_s, y_s, x_t):
        with tf.GradientTape() as tape:
            # class loss
            f_s = self.feature_extractor(x_s, training=True)
            preds = self.classifier(f_s, training=True)
            class_loss = self.classification_loss(y_s, preds)

            # adversarial loss
            f_t = self.feature_extractor(x_t, training=True)
            d_pred_t = self.discriminator(f_t, training=False)
            # "fake" label = 0 (we want D to predict "source" on target features)
            fake_labels = tf.zeros_like(d_pred_t)
            adv_loss = self.discriminator_loss(fake_labels, d_pred_t)

            total_loss = class_loss + self.grl_layer.lmbda * adv_loss

        # gradients on feature_extractor + classifier only
        variables = self.feature_extractor.trainable_variables + self.classifier.trainable_variables
        grads = tape.gradient(total_loss, variables)
        self.f_optimizer.apply_gradients(zip(grads, variables))
        return total_loss, class_loss

class GradientReversalLayer(Layer):
    def __init__(self, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lmbda = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        @tf.custom_gradient
        def _reverse_gradient(x):
            def grad(dy):
                return -self.lmbda * dy
            return x, grad
        return _reverse_gradient(inputs)
