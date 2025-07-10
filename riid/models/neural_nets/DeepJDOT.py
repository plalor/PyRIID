import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel
from time import perf_counter as timer


class DeepJDOT(PyRIIDModel):
    """Classifier using DeepJDOT (Deep Joint Distribution Optimal Transport) domain adaptation."""
    def __init__(self, optimizer=None, source_model=None, ot_weight=1.0, sinkhorn_reg=0.1, 
                 num_sinkhorn_iters=10, jdot_alpha=1.0, jdot_beta=1.0, dropout=0,
                 metrics=None):
        """
        Args:
            optimizer: tensorflow optimizer or optimizer name
            source_model: Pretrained source model.
            ot_weight: Weight for the OT loss term.
            sinkhorn_reg: Entropic regularization parameter for Sinkhorn iterations.
            num_sinkhorn_iters: Number of iterations in the Sinkhorn algorithm.
            jdot_alpha: Weight for the feature-distance term in the cost matrix.
            jdot_beta: Weight for the classification loss term in the cost matrix.
            dropout: dropout rate to apply to the adapted model layers
            metrics: list of metric functions
        """
        super().__init__()
        self.optimizer = optimizer or Adam(learning_rate=0.001)
        self.ot_weight = ot_weight
        self.sinkhorn_reg = sinkhorn_reg
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.jdot_alpha = jdot_alpha
        self.jdot_beta = jdot_beta
        self.dropout = dropout
        if metrics:
            self.metrics = {getattr(metric, '__name__', str(metric)): metric for metric in metrics}
        else:
            self.metrics = {}

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
            raise ValueError("Empty spectra provided!")

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

        # Define DeepJDOT model
        self.model = Model(
            inputs=self.feature_extractor.input,
            outputs=self.classifier(self.feature_extractor.output)
        )
        self.model.compile(loss=self.classification_loss)

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs=model_outputs,
            normalization=source_ss.spectra_state,
        )

        # Training loop
        self.history = {"total_loss": [], "class_loss": [], "ot_loss": [], "src_val_loss": [], "tgt_val_loss": [], "ot_val_loss": []}
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

            total_loss_avg = tf.keras.metrics.Mean()
            class_loss_avg = tf.keras.metrics.Mean()
            ot_loss_avg = tf.keras.metrics.Mean()
            for step in range(steps_per_epoch):
                (x_s, y_s), x_t = next(it)
                total_loss, class_loss, ot_loss = self.train_step(x_s, y_s, x_t)
                total_loss_avg.update_state(total_loss)
                class_loss_avg.update_state(class_loss)
                ot_loss_avg.update_state(ot_loss)

            src_class_loss_avg = tf.keras.metrics.Mean()
            tgt_class_loss_avg = tf.keras.metrics.Mean()
            ot_val_loss_avg = tf.keras.metrics.Mean()
            
            src_metric_avgs = {name: tf.keras.metrics.Mean() for name in self.metrics}
            tgt_metric_avgs = {name: tf.keras.metrics.Mean() for name in self.metrics}
            
            for (x_s_val, y_s_val), (x_t_val, y_t_val) in val_dataset:
                y_s_pred = self.model(x_s_val, training=False)
                loss_s = self.classification_loss(y_s_val, y_s_pred)
                src_class_loss_avg.update_state(loss_s)
            
                y_t_pred = self.model(x_t_val, training=False)
                loss_t = self.classification_loss(y_t_val, y_t_pred)
                tgt_class_loss_avg.update_state(loss_t)

                total_loss, class_loss, ot_loss = self.compute_losses(x_s_val, y_s_val, x_t_val, training=False)
                ot_val_loss_avg.update_state(ot_loss)
                
                for metric_name, metric_fn in self.metrics.items():
                    src_metric = metric_fn(y_s_val.numpy(), y_s_pred.numpy())
                    tgt_metric = metric_fn(y_t_val.numpy(), y_t_pred.numpy())
                    src_metric_avgs[metric_name].update_state(src_metric)
                    tgt_metric_avgs[metric_name].update_state(tgt_metric)

            total_loss = total_loss_avg.result().numpy()
            class_loss = class_loss_avg.result().numpy()
            ot_loss = ot_loss_avg.result().numpy()
            src_val_loss = src_class_loss_avg.result().numpy()
            tgt_val_loss = tgt_class_loss_avg.result().numpy()
            ot_val_loss = ot_val_loss_avg.result().numpy()

            self.history["total_loss"].append(total_loss)
            self.history["class_loss"].append(class_loss)
            self.history["ot_loss"].append(ot_loss)
            self.history["src_val_loss"].append(src_val_loss)
            self.history["tgt_val_loss"].append(tgt_val_loss)
            self.history["ot_val_loss"].append(ot_val_loss)
            
            for metric in self.metrics:
                metric_name = getattr(metric, '__name__', str(metric))
                self.history[f"src_val_{metric_name}"].append(src_metric_avgs[metric].result().numpy())
                self.history[f"tgt_val_{metric_name}"].append(tgt_metric_avgs[metric].result().numpy())

            if verbose:
                print(f"Finished in {timer()-t1:.0f} seconds")
                print("  "
                      f"total_loss: {total_loss_avg.result():.3g} - "
                      f"class_loss: {class_loss_avg.result():.3g} - "
                      f"ot_loss: {ot_loss_avg.result():.3g} - "
                      f"src_val_loss: {src_val_loss:.3g} - "
                      f"tgt_val_loss: {tgt_val_loss:.3g} - "
                      f"ot_val_loss: {ot_val_loss:.3g}")

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

    @staticmethod
    def pairwise_squared_distance(a, b):
        """
        Compute pairwise squared Euclidean distances between rows of tensors a and b.
        
        Args:
            a: Tensor of shape (n, d).
            b: Tensor of shape (m, d).
            
        Returns:
            Tensor of shape (n, m) containing squared distances.
        """
        a_norm = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)  # shape (n, 1)
        b_norm = tf.reduce_sum(tf.square(b), axis=1, keepdims=True)  # shape (m, 1)
        dist = a_norm + tf.transpose(b_norm) - 2 * tf.matmul(a, b, transpose_b=True)
        return tf.maximum(dist, 0.0)

    @staticmethod
    def pairwise_classification_loss(y_true, y_pred):
        """
        Compute pairwise classification loss (categorical crossentropy) between y_true and y_pred.
        y_true: Tensor of shape (n, num_classes)
        y_pred: Tensor of shape (m, num_classes)
        Returns: Tensor of shape (n, m)
        """
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        # Expand dims to compute loss for every source-target pair.
        y_true_exp = tf.expand_dims(y_true, axis=1)  # (n, 1, num_classes)
        y_pred_exp = tf.expand_dims(y_pred, axis=0)  # (1, m, num_classes)
        # Compute crossentropy: -sum(y_true * log(y_pred)) for each pair.
        ce = -tf.reduce_sum(y_true_exp * tf.math.log(y_pred_exp), axis=-1)  # shape (n, m)
        return ce

    @staticmethod
    def sinkhorn(a, b, M, reg, num_iters):
        """
        Compute the Sinkhorn algorithm for optimal transport.
        a: Tensor of shape (n,), source distribution (sums to 1)
        b: Tensor of shape (m,), target distribution (sums to 1)
        M: Cost matrix of shape (n, m)
        reg: Regularization parameter (epsilon)
        num_iters: Number of Sinkhorn iterations
        Returns: Optimal transport plan gamma of shape (n, m)
        """
        K = tf.exp(-M / reg)  # (n, m)
        u = tf.ones_like(a)
        v = tf.ones_like(b)

        for _ in range(num_iters):
            u = a / (tf.reduce_sum(K * tf.expand_dims(v, 0), axis=1) + 1e-8)
            v = b / (tf.reduce_sum(tf.transpose(K) * tf.expand_dims(u, 0), axis=1) + 1e-8)

        gamma = tf.expand_dims(u, 1) * K * tf.expand_dims(v, 0)
        return gamma

    @tf.function
    def compute_losses(self, x_s, y_s, x_t, training):
        # 1) forward passes
        f_s = self.feature_extractor(x_s, training=training)
        p_s = self.classifier(f_s, training=training)
        f_t = self.feature_extractor(x_t, training=training)
        p_t = self.classifier(f_t, training=training)

        # 2) classification loss on source
        class_loss = self.classification_loss(y_s, p_s)

        # 3) build cost matrix
        feat_dist  = self.pairwise_squared_distance(f_s, f_t)
        cls_matrix = self.pairwise_classification_loss(y_s, p_t)
        M = self.jdot_alpha * feat_dist + self.jdot_beta * cls_matrix

        # 4) OT coupling
        n = tf.shape(x_s)[0]; m = tf.shape(x_t)[0]
        a = tf.fill([n], 1.0/tf.cast(n,tf.float32))
        b = tf.fill([m], 1.0/tf.cast(m,tf.float32))
        gamma = self.sinkhorn(a, b, M, self.sinkhorn_reg, self.num_sinkhorn_iters)

        # 5) OT loss + total
        ot_loss    = tf.reduce_sum(gamma * M)
        total_loss = class_loss + self.ot_weight * ot_loss

        return total_loss, class_loss, ot_loss

    @tf.function
    def train_step(self, x_s, y_s, x_t):
        """
        Perform one training step:
          - Compute source features and predictions.
          - Compute target features and predictions.
          - Build a cost matrix that combines feature distances and a crossentropy term.
          - Compute the OT plan via Sinkhorn iterations.
          - Compute the OT loss and add it to the source classification loss.
        """
        with tf.GradientTape() as tape:
            total_loss, class_loss, ot_loss = self.compute_losses(x_s, y_s, x_t, training=True)
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss, class_loss, ot_loss
