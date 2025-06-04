import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Activation
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel
from time import perf_counter as timer


class DeepCORAL(PyRIIDModel):
    """Classifier using DeepCORAL domain adaptation."""
    def __init__(self, optimizer=None, source_model=None, lmbda=1):
        """
        Args:
            optimizer: tensorflow optimizer or optimizer name
            source_model: pretrained source model
            lmbda: weight for the CORAL loss
        """
        super().__init__()

        self.optimizer = optimizer or Adam(learning_rate=0.001)
        self.lmbda = lmbda

        if source_model is not None:
            self.classification_loss = source_model.loss

            # Remove dropout layers for stability
            def strip_dropout(layer):
                if isinstance(layer, Dropout):
                    return Activation('linear', name=layer.name)
                return layer.__class__.from_config(layer.get_config())
            
            self.source_model = clone_model(
                source_model,
                clone_function=strip_dropout
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
            batch_size=64, epochs=None, patience=None, target_level="Isotope", verbose=False, training_time=None):
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
            `tf.History` object.

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
        
        # Define CORAL model
        self.model = Model(
            inputs=self.feature_extractor.input,
            outputs=self.classifier(self.feature_extractor.output)
        )
        self.model.compile(loss = self.classification_loss)

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs=model_outputs,
            normalization=source_ss.spectra_state,
        )

        # Training loop
        self.history = {"total_loss": [], "class_loss": [], "coral_loss": [], "src_val_loss": [], "tgt_val_loss": [], "coral_val_loss": []}
        best_val_loss = np.inf
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
            coral_loss_avg = tf.keras.metrics.Mean()
            
            for step in range(steps_per_epoch):
                (x_s, y_s), x_t = next(it)
                total_loss, class_loss, coral_val = self.train_step(x_s, y_s, x_t)
                total_loss_avg.update_state(total_loss)
                class_loss_avg.update_state(class_loss)
                coral_loss_avg.update_state(coral_val)

            src_class_loss_avg = tf.keras.metrics.Mean()
            tgt_class_loss_avg = tf.keras.metrics.Mean()
            coral_val_loss_avg = tf.keras.metrics.Mean()
            for (x_s_val, y_s_val), (x_t_val, y_t_val) in val_dataset:
                y_s_pred = self.model(x_s_val, training=False)
                loss_s  = self.classification_loss(y_s_val, y_s_pred)
                src_class_loss_avg.update_state(loss_s)
            
                y_t_pred = self.model(x_t_val, training=False)
                loss_t  = self.classification_loss(y_t_val, y_t_pred)
                tgt_class_loss_avg.update_state(loss_t)
                
                f_s_val = self.feature_extractor(x_s_val, training=False)
                f_t_val = self.feature_extractor(x_t_val, training=False)
                coral_val_loss = self.coral_loss(f_s_val, f_t_val)
                coral_val_loss_avg.update_state(coral_val_loss)

            total_loss = total_loss_avg.result().numpy()
            class_loss = class_loss_avg.result().numpy()
            coral_loss = coral_loss_avg.result().numpy()

            src_val_loss = src_class_loss_avg.result().numpy()
            tgt_val_loss = tgt_class_loss_avg.result().numpy()
            coral_val_loss = coral_val_loss_avg.result().numpy()

            self.history["total_loss"].append(total_loss)
            self.history["class_loss"].append(class_loss)
            self.history["coral_loss"].append(coral_loss)
            self.history["src_val_loss"].append(src_val_loss)
            self.history["tgt_val_loss"].append(tgt_val_loss)
            self.history["coral_val_loss"].append(coral_val_loss)

            if verbose:
                print(f"Finished in {timer()-t1:.0f} seconds")
                print("  "
                      f"total_loss: {total_loss:.3g} - "
                      f"class_loss: {class_loss:.3g} - "
                      f"coral_loss: {coral_loss:.3g} - "
                      f"src_val_loss: {src_val_loss:.3g} - "
                      f"tgt_val_loss: {tgt_val_loss:.3g} - "
                      f"coral_val_loss: {coral_val_loss:.3g}")

            # Save best model weights based on the validation loss
            if tgt_val_loss < best_val_loss:
                best_val_loss = tgt_val_loss
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
    def coral_loss(source_features, target_features):
        """
        Computes CORAL loss between source and target features.
        Both are [batch_size, feature_dim] Tensors.
        """
        epsilon = 1e-8
        # 1) per‐feature zero‐mean, unit‐std
        s_mean = tf.reduce_mean(source_features, axis=0, keepdims=True)
        s_std  = tf.math.reduce_std(source_features, axis=0, keepdims=True) + epsilon
        t_mean = tf.reduce_mean(target_features, axis=0, keepdims=True)
        t_std  = tf.math.reduce_std(target_features, axis=0, keepdims=True) + epsilon

        s_norm = (source_features - s_mean) / s_std
        t_norm = (target_features - t_mean) / t_std

        # 2) covariance matrices on normalized features
        n_s = tf.cast(tf.shape(s_norm)[0], tf.float32)
        n_t = tf.cast(tf.shape(t_norm)[0], tf.float32)
        cov_s = tf.matmul(s_norm, s_norm, transpose_a=True) / (n_s - 1)
        cov_t = tf.matmul(t_norm, t_norm, transpose_a=True) / (n_t - 1)

        # 3) Normalize by 4d^2
        d = tf.cast(tf.shape(s_norm)[1], tf.float32)
        frob_sq = tf.reduce_sum(tf.square(cov_s - cov_t))
        return frob_sq / (4 * d**2)

    @tf.function
    def train_step(self, x_s, y_s, x_t):
        with tf.GradientTape() as tape:
            f_s = self.feature_extractor(x_s, training=True)
            preds_s = self.classifier(f_s, training=True)
            class_loss = self.classification_loss(y_s, preds_s)

            f_t = self.feature_extractor(x_t, training=True)
            coral_val = self.coral_loss(f_s, f_t)
            total_loss = class_loss + self.lmbda * coral_val

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss, class_loss, coral_val
