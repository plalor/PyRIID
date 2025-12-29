import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel
from riid.models.functions import modify_dropout_rate, clone_optimizer
from time import perf_counter as timer


class DAN(PyRIIDModel):
    """Classifier using Deep Adaptation Networks (DAN) for domain adaptation via 
    Maximum Mean Discrepancy  (MMD)."""
    def __init__(self, optimizer=None, source_model=None, lmbda=1, sigma=1.0,
                 kernel_num=11, kernel_mul=np.sqrt(2), dropout=0,
                 metrics=None):
        """
        Args:
            optimizer: tensorflow optimizer or optimizer name
            source_model: pretrained source model
            lmbda: weight for the MMD loss
            sigma: base bandwidth for the Gaussian kernel
            kernel_num: number of RBF kernels in the bank
            kernel_mul: geometric spacing between successive kernels
            dropout: dropout rate to apply to the adapted model layers
            metrics: list of metric functions
        """
        super().__init__()

        self.optimizer = optimizer or Adam(learning_rate=0.001)
        self.lmbda = lmbda
        self.base_sigma = sigma
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.dropout = dropout
        if metrics:
            self.metrics = {(getattr(m, "name", None) or getattr(m, "__name__", None) or str(m)): m for m in metrics}
        else:
            self.metrics = {}

        if source_model is not None:
            self.classification_loss = source_model.loss

            self.source_model = clone_model(
                source_model,
                clone_function=lambda layer: modify_dropout_rate(layer, self.dropout)
            )
            # self.source_model.build(source_model.input_shape)
            self.source_model.set_weights(source_model.get_weights())
            self.source_model.compile(
                optimizer=clone_optimizer(source_model.optimizer),
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
            batch_size=64, epochs=None, patience=None, es_mode="min", es_monitor="tgt_val_loss", target_level="Isotope", 
            validations_per_epoch=1, verbose=False, training_time=None):
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
            patience: number of validation checks to wait before early stopping
            es_mode: 'min' for loss-like metrics, 'max' for accuracy-like metrics
            target_level: `SampleSet.sources` column level to use
            validations_per_epoch: number of times to run validation per epoch (default: 1)
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

        center = self.kernel_num // 2
        self.sigma_list = [self.base_sigma * (self.kernel_mul ** (i - center)) for i in range(self.kernel_num)]
        self.sigma_list = [tf.constant(s, dtype=tf.float32) for s in self.sigma_list]

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

        # Define DAN model
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

        # Calculate steps between validations
        steps_between_validations = steps_per_epoch // validations_per_epoch
        
        # Training loop
        self.history = {"total_loss": [], "class_loss": [], "mmd_loss": [], "src_val_loss": [], "tgt_val_loss": [], "mmd_val_loss": []}
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
        early_stop = False
        
        while True:
            epoch += 1
            if epochs is not None and epoch > epochs:
                break
            
            # Loop through validations within this epoch
            for val_idx in range(validations_per_epoch):
                if verbose:
                    t1 = timer()
                    if epochs:
                        if validations_per_epoch > 1:
                            print(f"Epoch {epoch}/{epochs} - Validation {val_idx+1}/{validations_per_epoch}...", end="")
                        else:
                            print(f"Epoch {epoch}/{epochs}...", end="")
                    else:
                        if validations_per_epoch > 1:
                            print(f"Epoch {epoch} - Validation {val_idx+1}/{validations_per_epoch}...", end="")
                        else:
                            print(f"Epoch {epoch}...", end="")

                total_loss_avg = tf.keras.metrics.Mean()
                class_loss_avg = tf.keras.metrics.Mean()
                mmd_loss_avg = tf.keras.metrics.Mean()

                for step in range(steps_between_validations):
                    (x_s, y_s), x_t = next(it)
                    total_loss, class_loss, mmd_val = self.train_step(x_s, y_s, x_t)
                    total_loss_avg.update_state(total_loss)
                    class_loss_avg.update_state(class_loss)
                    mmd_loss_avg.update_state(mmd_val)

                src_class_loss_avg = tf.keras.metrics.Mean()
                tgt_class_loss_avg = tf.keras.metrics.Mean()
                mmd_val_loss_avg = tf.keras.metrics.Mean()
                
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
                    mmd_val_loss = self.mmd_loss(f_s_val, f_t_val)
                    mmd_val_loss_avg.update_state(mmd_val_loss)
                    
                    for metric_name, metric_fn in self.metrics.items():
                        src_metric = metric_fn(y_s_val.numpy(), y_s_pred.numpy())
                        tgt_metric = metric_fn(y_t_val.numpy(), y_t_pred.numpy())
                        src_metric_avgs[metric_name].update_state(src_metric)
                        tgt_metric_avgs[metric_name].update_state(tgt_metric)

                total_loss = total_loss_avg.result().numpy()
                class_loss = class_loss_avg.result().numpy()
                mmd_loss = mmd_loss_avg.result().numpy()
                src_val_loss = src_class_loss_avg.result().numpy()
                tgt_val_loss = tgt_class_loss_avg.result().numpy()
                mmd_val_loss = mmd_val_loss_avg.result().numpy()

                self.history["total_loss"].append(total_loss)
                self.history["class_loss"].append(class_loss)
                self.history["mmd_loss"].append(mmd_loss)
                self.history["src_val_loss"].append(src_val_loss)
                self.history["tgt_val_loss"].append(tgt_val_loss)
                self.history["mmd_val_loss"].append(mmd_val_loss)
                
                for metric in self.metrics:
                    metric_name = getattr(metric, '__name__', str(metric))
                    self.history[f"src_val_{metric_name}"].append(src_metric_avgs[metric].result().numpy())
                    self.history[f"tgt_val_{metric_name}"].append(tgt_metric_avgs[metric].result().numpy())

                if verbose:
                    print(f"finished in {timer()-t1:.0f} seconds")
                    print("  "
                          f"total_loss: {total_loss:.3g} - "
                          f"class_loss: {class_loss:.3g} - "
                          f"mmd_loss: {mmd_loss:.3g} - "
                          f"src_val_loss: {src_val_loss:.3g} - "
                          f"tgt_val_loss: {tgt_val_loss:.3g} - "
                          f"mmd_val_loss: {mmd_val_loss:.3g}")

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
                            print(f"No improvement for {patience} validation checks, stopping early.")
                        early_stop = True
                        break

                if timer() - t0 > training_time:
                    if verbose:
                        print("Reached preallotted training time, terminating.")
                    early_stop = True
                    break
            
            if early_stop:
                break

        if best_weights is not None:
            self.model.set_weights(best_weights)

        return self.history

    def predict(self, ss: SampleSet, batch_size: int = 1000):
        """Classify the spectra in the provided SampleSet.

        Results are stored inside the SampleSet's prediction-related properties.

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are either
                foreground (AKA, "net") or gross
            batch_size: batch size during call to self.model.predict
        """
        X = ss.get_samples().astype("float32")

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
    def gaussian_kernel(x, y, sigma):
        """
        Compute the Gaussian kernel matrix between x and y.
        Args:
            x: Tensor of shape [n, d].
            y: Tensor of shape [m, d].
            sigma: Bandwidth of the Gaussian kernel.
        Returns:
            A [n, m] kernel matrix.
        """
        x_norm = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        y_norm = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
        dist = x_norm - 2 * tf.matmul(x, y, transpose_b=True) + tf.transpose(y_norm)
        return tf.exp(-dist / (2.0 * sigma**2))
    
    def mmd_loss(self, source_features, target_features):
        """Multi-kernel MMD: average over RBFs with bandwidths in self.sigma_list"""
        def mean_no_diag(K):
            n = tf.shape(K)[0]
            mask = tf.ones_like(K) - tf.eye(n, dtype=K.dtype)
            return tf.reduce_sum(K * mask) / tf.cast(n * (n - 1), K.dtype)
        mmd_total = 0.0
        for sigma in self.sigma_list:
            K_ss = self.gaussian_kernel(source_features, source_features, sigma)
            K_tt = self.gaussian_kernel(target_features, target_features, sigma)
            K_st = self.gaussian_kernel(source_features, target_features, sigma)
            mmd_total += mean_no_diag(K_ss) + mean_no_diag(K_tt) - 2.0 * tf.reduce_mean(K_st)
        return mmd_total / len(self.sigma_list)

    @tf.function
    def train_step(self, x_s, y_s, x_t):
        with tf.GradientTape() as tape:
            f_s = self.feature_extractor(x_s, training=True)
            preds_s = self.classifier(f_s, training=True)
            class_loss = self.classification_loss(y_s, preds_s)

            f_t = self.feature_extractor(x_t, training=True)
            mmd_val = self.mmd_loss(f_s, f_t)
            total_loss = class_loss + self.lmbda * mmd_val

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss, class_loss, mmd_val
