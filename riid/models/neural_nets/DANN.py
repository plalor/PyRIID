import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType, SpectraState, read_hdf
from riid.models.base import ModelInput, PyRIIDModel
from riid.metrics import APE_score
from sklearn.metrics import accuracy_score
from time import perf_counter as time


class DANN(PyRIIDModel):
    """Domain Adversarial Neural Network classifier."""
    def __init__(self, optimizer=None, source_model=None, grl_layer_size=0,
                 gamma=10, lmbda=0):
        """
        Args:
            optimizer: tensorflow optimizer or optimizer name to use for training
            source_model: pretrained source model
            grl_layer_size: size of the gradient reversal dense layer
            gamma: hyperparameter for adjusting domain adaptation parameter
            lmbda: domain adaptation parameter. Ignored if gamma is specified
        """
        super().__init__()

        self.optimizer = optimizer
        self.grl_layer_size = grl_layer_size
        self.gamma = gamma
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
            batch_size: int = 200, epochs: int = 20, callbacks = None, patience: int = 10**4, 
            es_monitor: str = "val_ape_score", es_mode: str = "max", es_verbose=0, 
            target_level="Isotope", verbose: bool = False):
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
            callbacks: list of callbacks to be passed to the TensorFlow `Model.fit()` method
            patience: number of epochs to wait for `EarlyStopping` object
            es_monitor: quantity to be monitored for `EarlyStopping` object
            es_mode: mode for `EarlyStopping` object
            es_verbose: verbosity level for `EarlyStopping` object
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
                
        # Class labels (set dummy labels for target domain)
        source_contributions_df = source_ss.sources.T.groupby(target_level, sort=False).sum().T
        model_outputs = source_contributions_df.columns.values.tolist()
        class_labels_source = source_contributions_df.values.astype("float32")
        dummy_labels = np.zeros((len(X_target), class_labels_source.shape[1]))
        class_labels_target = dummy_labels.astype("float32")
        
        # Domain labels: 0 for source, 1 for target
        domain_labels_source = np.zeros((len(X_source), 1), dtype=np.float32)
        domain_labels_target = np.ones((len(X_target), 1), dtype=np.float32)

        # Make datasets
        def merge_batches(source_batch, target_batch):
            X_s, y_s_cls, y_s_dom = source_batch
            X_t, y_t_cls, y_t_dom = target_batch

            X = tf.concat([X_s, X_t], axis=0)
            class_labels = tf.concat([y_s_cls, y_t_cls], axis=0)
            domain_labels = tf.concat([y_s_dom, y_t_dom], axis=0)
            labels_dict = {"classifier": class_labels, "discriminator": domain_labels}

            batch_size_s = tf.shape(X_s)[0]
            batch_size_t = tf.shape(X_t)[0]
            weights_classifier = tf.concat([
                tf.ones((batch_size_s,), dtype=tf.float32),
                tf.zeros((batch_size_t,), dtype=tf.float32)
            ], axis=0)
            weights_discriminator = tf.ones((batch_size_s + batch_size_t,), dtype=tf.float32)

            weights_dict = {
                "classifier": weights_classifier,
                "discriminator": weights_discriminator
            }

            return X, labels_dict, weights_dict

        half_batch_size = batch_size // 2
        source_dataset = tf.data.Dataset.from_tensor_slices((X_source, class_labels_source, domain_labels_source))
        source_dataset = source_dataset.shuffle(len(X_source)).batch(half_batch_size)
        target_dataset = tf.data.Dataset.from_tensor_slices((X_target, class_labels_target, domain_labels_target))
        target_dataset = target_dataset.shuffle(len(X_target)).batch(half_batch_size)
        combined_dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
        combined_dataset = combined_dataset.map(merge_batches)

        if not self.model:
            inputs = self.feature_extractor.input
            features = self.feature_extractor(inputs)
            class_labels = self.classifier(features)

            self.grl_layer = GradientReversalLayer()
            grl = self.grl_layer(features)
            if self.grl_layer_size:
                grl = Dense(self.grl_layer_size, activation="relu")(grl)
            domain_labels = Dense(1, activation="sigmoid", name="discriminator")(grl)

            self.model = Model(inputs=inputs, outputs={
                "classifier": class_labels,
                "discriminator": domain_labels
            })

            self.model.compile(
                loss={"classifier": self.classification_loss, "discriminator": "binary_crossentropy"},
                optimizer=self.optimizer,
            )

        self._update_info(
            target_level=target_level,
            model_outputs = source_ss.sources.T.groupby(target_level, sort=False).sum().T.columns.values.tolist(),
            normalization=source_ss.spectra_state,
        )

        es = EarlyStopping(
            monitor=es_monitor,
            patience=patience,
            verbose=es_verbose,
            restore_best_weights=True,
            mode=es_mode,
        )
        vc = ValidationCallback(self, validation_ss, target_level, batch_size)
        if callbacks:
            callbacks.append(vc)
        else:
            callbacks = [vc]
        if self.gamma > 0:
            lambda_scheduler = LambdaScheduler(gamma=self.gamma, total_epochs=epochs, grl_layer=self.grl_layer)
            callbacks.append(lambda_scheduler)
        else:
            tf.keras.backend.set_value(self.grl_layer.lmbda, self.lmbda)
        callbacks.append(es)

        t0 = time()
        history = self.model.fit(
            combined_dataset,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )
        self.history = history.history
        self.history["training_time"] = time() - t0

        return history

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

        results = self.model.predict(X, batch_size=batch_size)["classifier"]

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
        """Calculates the label predictor loss on ss"""
        self.predict(ss, batch_size=batch_size)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        loss = self.model.loss["classifier"](y_true, y_pred).numpy()
        return loss

    def calc_domain_accuracy(self, ss: SampleSet, domain_label: int, batch_size: int = 1000):
        """Calculate the domain classification accuracy on ss"""
        x_test = ss.get_samples().astype(float)
        preds = self.model.predict(x_test, batch_size=batch_size)
        domain_preds = np.round(preds["discriminator"]).astype(int).flatten()
        domain_labels = np.full(shape=(len(x_test),), fill_value=domain_label, dtype=int)
        accuracy = accuracy_score(domain_labels, domain_preds)
        return accuracy


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


class LambdaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, gamma, total_epochs, grl_layer):
        super().__init__()
        self.gamma = gamma
        self.total_epochs = total_epochs
        self.grl_layer = grl_layer

    def on_epoch_begin(self, epoch, logs=None):
        p = epoch / self.total_epochs
        new_lmbda = 2 / (1 + np.exp(-self.gamma * p)) - 1
        tf.keras.backend.set_value(self.grl_layer.lmbda, new_lmbda)


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, DANN_instance, validation_ss, target_level="Isotope", batch_size: int = 1000):
        super().__init__()
        self.DANN_instance = DANN_instance
        self.validation_ss = validation_ss
        self.target_level = target_level
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        ape_score = self.DANN_instance.calc_APE_score(self.validation_ss, target_level=self.target_level, batch_size=self.batch_size)
        logs["val_ape_score"] = ape_score
