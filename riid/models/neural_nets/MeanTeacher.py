import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel
from riid.models.functions import modify_dropout_rate, clone_optimizer, poisson_resample
from time import perf_counter as timer


class MeanTeacher(PyRIIDModel):
    """Classifier using Mean Teacher (EMA teacher) for domain adaptation."""
    def __init__(self, optimizer=None, source_model=None, consistency_weight=1.0, 
                 ema_decay=0.999, effective_counts=0, dropout=0, metrics=None):
        """
        Args:
            optimizer: tensorflow optimizer or optimizer name for student model
            source_model: pretrained source model
            consistency_weight: weight for the consistency loss
            ema_decay: exponential moving average decay rate for teacher updates
            effective_counts: effective counts for Poisson resampling
            dropout: dropout rate to apply to the student model
            metrics: list of metric functions
        """
        super().__init__()

        self.optimizer = optimizer or Adam(learning_rate=0.001)
        self.consistency_weight = consistency_weight
        self.ema_decay = ema_decay
        self.dropout = dropout
        self.effective_counts = effective_counts
        self.consistency_loss = MeanSquaredError()
        if metrics:
            self.metrics = {(getattr(m, "name", None) or getattr(m, "__name__", None) or str(m)): m for m in metrics}
        else:
            self.metrics = {}

        if source_model is not None:
            self.classification_loss = source_model.loss

            # Create student model with dropout
            self.student_model = clone_model(
                source_model,
                clone_function=lambda layer: modify_dropout_rate(layer, self.dropout)
            )
            self.student_model.set_weights(source_model.get_weights())
            self.student_model.compile(
                optimizer=clone_optimizer(source_model.optimizer),
                loss=source_model.loss,
                metrics=source_model.metrics
            )

            # Create teacher model without dropout (for stability)
            self.teacher_model = clone_model(
                source_model,
                clone_function=lambda layer: modify_dropout_rate(layer, 0.0)
            )
            self.teacher_model.set_weights(source_model.get_weights())
            self.teacher_model.compile(
                optimizer=clone_optimizer(source_model.optimizer),
                loss=source_model.loss,
                metrics=source_model.metrics
            )

            # Extract student feature extractor and classifier
            student_layers = self.student_model.layers
            student_fe_input = self.student_model.input
            student_fe_output = student_layers[-2].output
            self.student_feature_extractor = Model(
                inputs=student_fe_input, 
                outputs=student_fe_output, 
                name="student_feature_extractor"
            )

            student_classifier_input = Input(
                shape=student_fe_output.shape[1:], 
                name="student_encoder_output"
            )
            student_classifier_output = student_layers[-1](student_classifier_input)
            self.student_classifier = Model(
                inputs=student_classifier_input, 
                outputs=student_classifier_output, 
                name="student_classifier"
            )

            # Extract teacher feature extractor and classifier
            teacher_layers = self.teacher_model.layers
            teacher_fe_input = self.teacher_model.input
            teacher_fe_output = teacher_layers[-2].output
            self.teacher_feature_extractor = Model(
                inputs=teacher_fe_input, 
                outputs=teacher_fe_output, 
                name="teacher_feature_extractor"
            )

            teacher_classifier_input = Input(
                shape=teacher_fe_output.shape[1:], 
                name="teacher_encoder_output"
            )
            teacher_classifier_output = teacher_layers[-1](teacher_classifier_input)
            self.teacher_classifier = Model(
                inputs=teacher_classifier_input, 
                outputs=teacher_classifier_output, 
                name="teacher_classifier"
            )
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
            es_monitor: metric to monitor for early stopping
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

        # Define student model (used for final predictions via teacher)
        self.model = Model(
            inputs=self.student_feature_extractor.input,
            outputs=self.student_classifier(self.student_feature_extractor.output)
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
        self.history = {
            "total_loss": [], 
            "class_loss": [], 
            "consistency_loss": [], 
            "src_val_loss": [], 
            "tgt_val_loss": [], 
            "consistency_val_loss": []
        }
        for metric_name in self.metrics:
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
                consistency_loss_avg = tf.keras.metrics.Mean()

                for step in range(steps_between_validations):
                    (x_s, y_s), (x_t,) = next(it)
                    total_loss, class_loss, consistency_val = self.train_step(x_s, y_s, x_t)
                    self.update_teacher_weights()
                    total_loss_avg.update_state(total_loss)
                    class_loss_avg.update_state(class_loss)
                    consistency_loss_avg.update_state(consistency_val)

                src_class_loss_avg = tf.keras.metrics.Mean()
                tgt_class_loss_avg = tf.keras.metrics.Mean()
                consistency_val_loss_avg = tf.keras.metrics.Mean()
                
                src_metric_avgs = {name: tf.keras.metrics.Mean() for name in self.metrics}
                tgt_metric_avgs = {name: tf.keras.metrics.Mean() for name in self.metrics}
                
                for (x_s_val, y_s_val), (x_t_val, y_t_val) in val_dataset:
                    # Use teacher model for validation predictions (more stable)
                    y_s_pred_teacher = self.teacher_model(x_s_val, training=False)
                    loss_s = self.classification_loss(y_s_val, y_s_pred_teacher)
                    src_class_loss_avg.update_state(loss_s)
                    
                    y_t_pred_teacher = self.teacher_model(x_t_val, training=False)
                    loss_t = self.classification_loss(y_t_val, y_t_pred_teacher)
                    tgt_class_loss_avg.update_state(loss_t)
                    
                    # Consistency loss between student and teacher on target data
                    y_t_pred_student = self.student_model(x_t_val, training=False)
                    consistency_val_loss = self.consistency_loss(y_t_pred_teacher, y_t_pred_student)
                    consistency_val_loss_avg.update_state(consistency_val_loss)
                    
                    for metric_name, metric_fn in self.metrics.items():
                        src_metric = metric_fn(y_s_val.numpy(), y_s_pred_teacher.numpy())
                        tgt_metric = metric_fn(y_t_val.numpy(), y_t_pred_teacher.numpy())
                        src_metric_avgs[metric_name].update_state(src_metric)
                        tgt_metric_avgs[metric_name].update_state(tgt_metric)

                total_loss = total_loss_avg.result().numpy()
                class_loss = class_loss_avg.result().numpy()
                consistency_loss = consistency_loss_avg.result().numpy()
                src_val_loss = src_class_loss_avg.result().numpy()
                tgt_val_loss = tgt_class_loss_avg.result().numpy()
                consistency_val_loss = consistency_val_loss_avg.result().numpy()

                self.history["total_loss"].append(total_loss)
                self.history["class_loss"].append(class_loss)
                self.history["consistency_loss"].append(consistency_loss)
                self.history["src_val_loss"].append(src_val_loss)
                self.history["tgt_val_loss"].append(tgt_val_loss)
                self.history["consistency_val_loss"].append(consistency_val_loss)
                
                for metric_name in self.metrics:
                    self.history[f"src_val_{metric_name}"].append(src_metric_avgs[metric_name].result().numpy())
                    self.history[f"tgt_val_{metric_name}"].append(tgt_metric_avgs[metric_name].result().numpy())

                if verbose:
                    print(f"Finished in {timer()-t1:.0f} seconds")
                    print("  "
                          f"total_loss: {total_loss:.3g} - "
                          f"class_loss: {class_loss:.3g} - "
                          f"consistency_loss: {consistency_loss:.3g} - "
                          f"src_val_loss: {src_val_loss:.3g} - "
                          f"tgt_val_loss: {tgt_val_loss:.3g} - "
                          f"consistency_val_loss: {consistency_val_loss:.3g}")

                current_metric = self.history[es_monitor][-1]
                is_better = current_metric < best_metric if es_mode == "min" else current_metric > best_metric
                
                if is_better:
                    best_metric = current_metric
                    best_weights = self.teacher_model.get_weights()
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
            self.teacher_model.set_weights(best_weights)

        return self.history

    def predict(self, ss: SampleSet, batch_size: int = 1000):
        """Classify the spectra in the provided `SampleSet`.

        Results are stored inside the SampleSet's prediction-related properties.

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are either
                foreground (AKA, "net") or gross
            batch_size: batch size during call to self.model.predict
        """
        X = ss.get_samples().astype("float32")

        # Use teacher model for inference (more stable than student)
        results = self.teacher_model.predict(X, batch_size=batch_size)

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
    def train_step(self, x_s, y_s, x_t):
        """Train student model on source classification + target consistency."""
        with tf.GradientTape() as tape:
            # Classification loss on source data
            f_s = self.student_feature_extractor(x_s, training=True)
            preds_s = self.student_classifier(f_s, training=True)
            class_loss = self.classification_loss(y_s, preds_s)

            # Poisson resample target data for student and teacher
            x_t_student = poisson_resample(x_t, self.effective_counts)
            x_t_teacher = poisson_resample(x_t, self.effective_counts)

            # Consistency loss on target data (student vs teacher)
            student_preds_t = self.student_model(x_t_student, training=True)
            teacher_preds_t = self.teacher_model(x_t_teacher, training=False)
            consistency_loss = self.consistency_loss(teacher_preds_t, student_preds_t)

            total_loss = class_loss + self.consistency_weight * consistency_loss

        # Update only student model weights
        grads = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student_model.trainable_variables))
        
        return total_loss, class_loss, consistency_loss

    @tf.function
    def update_teacher_weights(self):
        """Update teacher model weights using EMA of student weights."""
        for t, s in zip(self.teacher_model.trainable_variables, self.student_model.trainable_variables):
            t.assign(self.ema_decay * t + (1.0 - self.ema_decay) * s)

        for t, s in zip(self.teacher_model.non_trainable_variables, self.student_model.non_trainable_variables):
            t.assign(s)
