import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType, SpectraState, read_hdf
from riid.models.base import ModelInput, PyRIIDModel
from riid.metrics import APE_score
from time import perf_counter as time


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

        self.optimizer = optimizer
        self.lmbda = lmbda

        if source_model is not None:
            self.classification_loss = source_model.loss

            # Remove dropout layers for stability
            config = source_model.get_config()
            filtered_layers = [layer for layer in config["layers"] if layer["class_name"] != "Dropout"]
            config["layers"] = filtered_layers
            self.source_model = Model.from_config(config)
            self.source_model.set_weights(source_model.get_weights())

            all_layers = self.source_model.layers
            feature_extractor_input = self.source_model.input
            feature_extractor_output = all_layers[-2].output
            self.feature_extractor = Model(inputs=feature_extractor_input, outputs=feature_extractor_output, name="feature_extractor")

            classifier_input = Input(shape=feature_extractor_output.shape[1:], name="feature_extractor_output")
            classifier_output = all_layers[-1](classifier_input)
            self.classifier = Model(inputs=classifier_input, outputs=classifier_output, name="classifier")
        else:
            print("WARNING: no pretrained source model was provided")

        if self.optimizer is None:
            self.optimizer = Adam(learning_rate=0.001)

        self.model = None

    def fit(self, source_ss: SampleSet, target_ss: SampleSet, validation_ss: SampleSet,
            batch_size: int = 200, epochs: int = 20, target_level="Isotope", verbose: bool = False):
        """Fit a model to the given `SampleSet`(s).

        Args:
            source_ss: `SampleSet` of `n` training spectra from the source domain where `n` >= 1
                and the spectra are either foreground (AKA, "net") or gross.
            target_ss: `SampleSet` of `n` training spectra from the target domain where `n` >= 1
                and the spectra are either foreground (AKA, "net") or gross.
            validation_ss: `SampleSet` of `m` validation spectra from the target domain where 
                `m` >= 1 and the spectra are either foreground (AKA, "net") or gross.
            batch_size: number of samples per gradient update
            epochs: maximum number of training iterations
            target_level: `SampleSet.sources` column level to use
            verbose: whether to show detailed model training output

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
                
        ### Preparing training data
        X_source = source_ss.get_samples().astype("float32")
        X_target = target_ss.get_samples().astype("float32")

        source_contributions_df = source_ss.sources.T.groupby(target_level, sort=False).sum().T
        model_outputs = source_contributions_df.columns.values.tolist()
        Y_source = source_contributions_df.values.astype("float32")

        # Make datasets
        source_dataset = tf.data.Dataset.from_tensor_slices((X_source, Y_source))
        source_dataset = source_dataset.shuffle(len(X_source)).batch(batch_size)
        target_dataset = tf.data.Dataset.from_tensor_slices((X_target))
        target_dataset = target_dataset.shuffle(len(X_target)).batch(batch_size)

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
        self.history = {"class_loss": [], "total_loss": [], "coral_loss": [], "val_ape_score": []}
        best_val_ape = -np.inf
        best_weights = None
        t0 = time()
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
                t1 = time()

            total_loss_avg = tf.metrics.Mean()
            class_loss_avg = tf.metrics.Mean()
            coral_loss_avg = tf.metrics.Mean()

            for (x_s, y_s), x_t in zip(source_dataset, target_dataset):
                total_loss, class_loss, coral_val = self.train_step(x_s, y_s, x_t)
                total_loss_avg.update_state(total_loss)
                class_loss_avg.update_state(class_loss)
                coral_loss_avg.update_state(coral_val)

            val_ape_score = self.calc_APE_score(validation_ss, target_level=target_level, batch_size=batch_size)
            self.history["class_loss"].append(float(class_loss_avg.result()))
            self.history["total_loss"].append(float(total_loss_avg.result()))
            self.history["coral_loss"].append(float(coral_loss_avg.result()))
            self.history["val_ape_score"].append(val_ape_score)

            # Save best model weights based on the validation APE score
            if val_ape_score > best_val_ape:
                best_val_ape = val_ape_score
                best_weights = self.model.get_weights()

            if verbose:
                print(f"Finished in {time()-t1:.0f} seconds")
                print("  "
                      f"total_loss: {total_loss_avg.result():.4f} - "
                      f"class_loss: {class_loss_avg.result():.4f} - "
                      f"coral_loss: {coral_loss_avg.result():.4f} - "
                      f"val_ape_score: {val_ape_score:.4f}")

        self.history["training_time"] = time() - t0
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

    def calc_APE_score(self, ss: SampleSet, target_level="Isotope", batch_size: int = 1000):
        """Calculate the prediction APE score on ss"""
        self.predict(ss, batch_size=batch_size)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        ape = APE_score(y_true, y_pred).numpy()
        return ape

    def calc_cosine_similarity(self, ss: SampleSet, target_level="Isotope", batch_size: int = 1000):
        """Calculate the prediction cosine similarity score on ss"""
        self.predict(ss, batch_size=batch_size)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        cosine_sim = cosine_similarity(y_true, y_pred)
        cosine_score = -tf.reduce_mean(cosine_sim).numpy()
        return cosine_score

    def calc_loss(self, ss: SampleSet, target_level="Isotope", batch_size: int = 1000):
        """Calculate the loss on ss"""
        self.predict(ss, batch_size=batch_size)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        loss = self.model.loss(y_true, y_pred).numpy()
        return loss

    @staticmethod
    def coral_loss(source_features, target_features):
        """
        Computes CORAL loss between source and target features.
        Both are [batch_size, feature_dim] Tensors.
        """
        s_mean = tf.reduce_mean(source_features, axis=0, keepdims=True)
        t_mean = tf.reduce_mean(target_features, axis=0, keepdims=True)
        s_centered = source_features - s_mean
        t_centered = target_features - t_mean

        cov_source = tf.matmul(tf.transpose(s_centered), s_centered)
        cov_target = tf.matmul(tf.transpose(t_centered), t_centered)

        return tf.reduce_mean(tf.square(cov_source - cov_target))

    @tf.function
    def train_step(self, x_s, y_s, x_t):
        with tf.GradientTape() as tape:
            f_s = self.feature_extractor(x_s, training=True)
            preds_s = self.classifier(f_s, training=True)
            class_loss = self.classification_loss(y_s, preds_s)

            f_t = self.feature_extractor(x_t, training=True)
            coral_val = self.coral_loss(f_s, f_t)
            total_loss = class_loss + self.lmbda * coral_val

        grads = tape.gradient(total_loss, self.source_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.source_model.trainable_variables))
        return total_loss, class_loss, coral_val
