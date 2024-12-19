import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten, Layer
from tensorflow.keras.losses import CategoricalCrossentropy, cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2

from riid import SampleSet, SpectraType, SpectraState, read_hdf
from riid.models.base import ModelInput, PyRIIDModel
from riid.metrics import APE_score
from sklearn.metrics import accuracy_score


class DANN(PyRIIDModel):
    """Domain Adversarial Neural Network classifier."""
    def __init__(self, activation=None, loss=None, optimizer=None,
                 metrics=None, l2_alpha: float = 1e-4,
                 activity_regularizer=None, final_activation=None,
                 hidden_layers=None, dense_layer_size=None, grl_layer_size=0,
                 gamma=10, dropout=0):
        """
        Args:
            activation: activation function to use for each dense layer
            loss: loss function to use for training
            optimizer: tensorflow optimizer or optimizer name to use for training
            metrics: list of metrics to be evaluating during training
            l2_alpha: alpha value for the L2 regularization of each dense layer
            activity_regularizer: regularizer function applied each dense layer output
            final_activation: final activation function to apply to model output
            hidden_layers: (filter, kernel_size) of each hidden laye of the CNN
            dense_layer_size: size of the final dense layer after the convolutional layers
            grl_layer_size: size of the gradient reversal dense layer
            gamma: hyperparameter for adjusting domain adaptation parameter
            dropout: optional droupout layer after each hidden layer
        """
        super().__init__()

        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.l2_alpha = l2_alpha
        self.activity_regularizer = activity_regularizer
        self.final_activation = final_activation

        self.hidden_layers = hidden_layers
        self.dense_layer_size = dense_layer_size
        self.grl_layer_size = grl_layer_size
        self.gamma = gamma
        self.dropout = dropout

        if self.activation is None:
            self.activation = "relu"
        if self.loss is None:
            self.loss = CategoricalCrossentropy()
        if optimizer is None:
            self.optimizer = Adam(learning_rate=0.001)
        if self.activity_regularizer is None:
            self.activity_regularizer = l1(0.0)
        if self.final_activation is None:
            self.final_activation = "softmax"

        self.model = None
        self._set_predict_fn()

    def fit(self, source_training_ss: SampleSet, target_training_ss: SampleSet, source_validation_ss: SampleSet, 
            target_validation_ss: SampleSet, batch_size: int = 200, epochs: int = 20, callbacks = None, 
            patience: int = 10, es_monitor: str = "val_target_ape_score", es_mode: str = "max", es_verbose=0,
            target_level="Isotope", verbose: bool = False):
        """Fit a model to the given `SampleSet`(s).

        Args:
            source_training_ss: `SampleSet` of `n` training spectra from the source domain where `n` >= 1 
                and the spectra are either foreground (AKA, "net") or gross.
            target_training_ss: `SampleSet` of `m` training spectra from the target domain where `m` >= 1
                and the spectra are either foreground (AKA, "net") or gross.
            source_validation_ss: `SampleSet` of source domain validation spectra.
            target_validation_ss: `SampleSet` of target domain validation spectra.
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

        if source_training_ss.n_samples <= 0:
            raise ValueError("No spectr[a|um] provided!")

        if source_training_ss.spectra_type == SpectraType.Gross:
            self.model_inputs = (ModelInput.GrossSpectrum,)
        elif source_training_ss.spectra_type == SpectraType.Foreground:
            self.model_inputs = (ModelInput.ForegroundSpectrum,)
        elif source_training_ss.spectra_type == SpectraType.Background:
            self.model_inputs = (ModelInput.BackgroundSpectrum,)
        else:
            raise ValueError(f"{source_training_ss.spectra_type} is not supported in this model.")

        ### Preparing training data
        X_source_train = source_training_ss.get_samples()
        X_target_train = target_training_ss.get_samples()
        X_train = np.concatenate((X_source_train, X_target_train)).astype("float32")
                
        # Label predictor labels (set dummy labels for target domain)
        source_contributions_df = source_training_ss.sources.T.groupby(target_level, sort=False).sum().T
        model_outputs = source_contributions_df.columns.values.tolist()
        predictor_labels_source_train = source_contributions_df.values
        dummy_labels = np.zeros((len(X_target_train), predictor_labels_source_train.shape[1]))
        # predictor_labels_target_train = target_training_ss.sources.T.groupby(target_level, sort=False).sum().T.values
        predictor_labels_train = np.concatenate((predictor_labels_source_train, dummy_labels)).astype("float32")
        
        # Domain labels: 0 for source, 1 for target
        domain_labels_source_train = np.zeros(len(X_source_train)).reshape(-1, 1)
        domain_labels_target_train = np.ones(len(X_target_train)).reshape(-1, 1)
        domain_labels_train = np.concatenate((domain_labels_source_train, domain_labels_target_train)).astype("float32")
        
        # Weights for label predictor (1 for source, 0 for target) and domain classifier (1 for all samples)
        weights_label_predictor_train = np.concatenate((np.ones(len(X_source_train)), np.zeros(len(X_target_train)))).astype("float32")
        weights_domain_classifier_train = np.ones(len(X_train)).astype("float32")

        # Shuffle the training data
        num_source = len(X_source_train)
        num_target = len(X_target_train)
        if num_source == num_target:
            print("Balanced source and target datasets, interweaving batches")
            half_batch = batch_size // 2
            source_indices = np.arange(num_source)
            target_indices = np.arange(num_source, num_source + num_target)
            np.random.shuffle(source_indices)
            np.random.shuffle(target_indices)
            num_batches = np.ceil((num_source + num_target) / batch_size).astype(int)
            indices = []
            for i in range(num_batches):
                batch_source = source_indices[i*half_batch : (i+1)*half_batch]
                batch_target = target_indices[i*half_batch : (i+1)*half_batch]
                batch_indices = np.concatenate([batch_source, batch_target])
                np.random.shuffle(batch_indices)
                indices.append(batch_indices)
            indices = np.concatenate(indices)
        else:
            print("WARNING: imbalanced source and target datasets")
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)

        assert len(indices) == len(X_train)
        X_train = X_train[indices]
        predictor_labels_train = predictor_labels_train[indices]
        domain_labels_train = domain_labels_train[indices]
        weights_label_predictor_train = weights_label_predictor_train[indices]
        weights_domain_classifier_train = weights_domain_classifier_train[indices]
        
        # Prepare label and weight dictionaries
        labels_dict_train = {
            "label_predictor": predictor_labels_train,
            "domain_classifier": domain_labels_train
        }
        sample_weight_train = {
            "label_predictor": weights_label_predictor_train,
            "domain_classifier": weights_domain_classifier_train
        }

        ### Preparing validation data
        X_source_validation = source_validation_ss.get_samples()
        X_target_validation = target_validation_ss.get_samples()
        X_validation = np.concatenate((X_source_validation, X_target_validation)).astype("float32")
        
        # Label predictor labels
        predictor_labels_source_validation = source_validation_ss.sources.T.groupby(target_level, sort=False).sum().T.values
        predictor_labels_target_validation = target_validation_ss.sources.T.groupby(target_level, sort=False).sum().T.values
        predictor_labels_validation = np.concatenate((predictor_labels_source_validation, predictor_labels_target_validation)).astype("float32")
        
        # Domain labels: 0 for source, 1 for target
        domain_labels_source_validation = np.zeros(len(X_source_validation)).reshape(-1, 1)
        domain_labels_target_validation = np.ones(len(X_target_validation)).reshape(-1, 1)
        domain_labels_validation = np.concatenate((domain_labels_source_validation, domain_labels_target_validation)).astype("float32")

        # Weights for label predictor (1 for source, 0 for target) and domain classifier (1 for all samples)
        weights_label_predictor_validation = np.concatenate((np.ones(len(X_source_validation)), np.zeros(len(X_target_validation)))).astype("float32")
        weights_domain_classifier_validation = np.ones(len(X_validation)).astype("float32")

        # Prepare label and weight dictionaries
        labels_dict_validation = {
            "label_predictor": predictor_labels_validation,
            "domain_classifier": domain_labels_validation
        }
        sample_weight_validation = {
            "label_predictor": weights_label_predictor_validation,
            "domain_classifier": weights_domain_classifier_validation
        }

        # Prepare validation data for callbacks
        source_validation_data = (X_source_validation, predictor_labels_source_validation, domain_labels_source_validation)
        target_validation_data = (X_target_validation, predictor_labels_target_validation, domain_labels_target_validation)
        ###
        
        if not self.model:
            input_shape = X_train.shape[1]
            inputs = Input(shape=(input_shape,1), name="Spectrum")
            if self.hidden_layers is None:
                self.hidden_layers = [(32, 5), (64, 3)]

            # Feature extractor
            x = inputs
            for layer, (filters, kernel_size) in enumerate(self.hidden_layers):
                x = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation=self.activation,
                    activity_regularizer=self.activity_regularizer,
                    kernel_regularizer=l2(self.l2_alpha),
                    name=f"conv_{layer}"
                )(x)
                x = MaxPooling1D(pool_size=2)(x)
                
                if self.dropout > 0:
                    x = Dropout(self.dropout)(x)

            x = Flatten()(x)
            if self.dense_layer_size is None:
                self.dense_layer_size = input_shape//2
            x = Dense(self.dense_layer_size, activation=self.activation)(x)
            if self.dropout > 0:
                x = Dropout(self.dropout)(x)

            # Label predictor
            label_outputs = Dense(predictor_labels_train.shape[1],
                                  activation=self.final_activation,
                                  name="label_predictor")(x)

            # Domain classifier branch with GRL
            grl_layer = GradientReversalLayer()
            grl = grl_layer(x)
            if self.grl_layer_size:
                grl = Dense(self.grl_layer_size, activation='relu')(grl)
            domain_output = Dense(1, activation="sigmoid", name="domain_classifier")(grl)

            # Combined model            
            self.model = Model(inputs, outputs={
                "label_predictor": label_outputs,
                "domain_classifier": domain_output
            })
            self.model.compile(
                loss={"label_predictor": self.loss, "domain_classifier": "binary_crossentropy"},
                optimizer=self.optimizer,
                metrics=self.metrics
            )

        es = EarlyStopping(
            monitor=es_monitor,
            patience=patience,
            verbose=es_verbose,
            restore_best_weights=True,
            mode=es_mode,
        )
        dual_validation_callback = DualValidationCallback(
            source_data=source_validation_data,
            target_data=target_validation_data,
            loss = self.model.loss,
            batch_size=batch_size,
        )
        lambda_scheduler = LambdaScheduler(gamma=self.gamma, total_epochs=epochs, grl_layer=grl_layer)

        if callbacks is None:
            callbacks = []
        callbacks.append(dual_validation_callback)
        callbacks.append(lambda_scheduler)
        callbacks.append(es)
        
        history = self.model.fit(
            x=X_train,
            y=labels_dict_train,
            sample_weight=sample_weight_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(X_validation, labels_dict_validation, sample_weight_validation),
            callbacks=callbacks,
            shuffle=False,
        )
        self.history = history.history

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs=model_outputs,
            normalization=source_training_ss.spectra_state,
        )

        # Define the predict function with tf.function and input_signature
        self._set_predict_fn()

        return history

    def _set_predict_fn(self):
        self._predict_fn = tf.function(
            self._predict,
            experimental_relax_shapes=True
        )

    def _predict(self, input_tensor):
        predictions = self.model(input_tensor, training=False)
        return predictions["label_predictor"]

    def predict(self, ss: SampleSet, bg_ss: SampleSet = None):
        """Classify the spectra in the provided `SampleSet`(s).

        Results are stored inside the first SampleSet's prediction-related properties.

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are either
                foreground (AKA, "net") or gross
            bg_ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are background
        """
        x_test = ss.get_samples().astype(float)
        if bg_ss:
            X = [x_test, bg_ss.get_samples().astype(float)]
        else:
            X = x_test

        spectra_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        results = self._predict_fn(spectra_tensor)

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

    def calc_APE_score(self, ss: SampleSet, target_level="Isotope"):
        """Calculate the prediction APE score on ss"""
        self.predict(ss)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        ape = APE_score(y_true, y_pred).numpy()
        return ape

    def calc_cosine_similarity(self, ss: SampleSet, target_level="Isotope"):
        """Calculate the prediction cosine similarity score on ss"""
        self.predict(ss)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        cosine_sim = cosine_similarity(y_true, y_pred)
        cosine_score = -tf.reduce_mean(cosine_sim).numpy()
        return cosine_score

    def calc_loss(self, ss: SampleSet, target_level="Isotope"):
        """Calculates the label predictor loss on ss"""
        self.predict(ss)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        loss = self.model.loss["label_predictor"](y_true, y_pred).numpy()
        return loss


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


class DualValidationCallback(tf.keras.callbacks.Callback):
    """Calculates metrics (ape score, cosine similarity, loss, domain accuracy)
    separately on the source and target domain"""
    def __init__(self, source_data, target_data, loss, batch_size=64):
        super().__init__()
        self.source_data = source_data
        self.target_data = target_data
        self.loss = loss
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
            
        source_metrics = self.evaluate_domain(self.source_data)
        target_metrics = self.evaluate_domain(self.target_data)

        logs['val_source_ape_score'] = source_metrics['ape_score']
        logs['val_source_cosine_similarity'] = source_metrics['cosine_similarity']
        logs['val_source_loss'] = source_metrics['loss']
        logs['val_source_domain_accuracy'] = source_metrics['domain_accuracy']
        
        logs['val_target_ape_score'] = target_metrics['ape_score']
        logs['val_target_cosine_similarity'] = target_metrics['cosine_similarity']
        logs['val_target_loss'] = target_metrics['loss']
        logs['val_target_domain_accuracy'] = target_metrics['domain_accuracy']

    def evaluate_domain(self, data):
        X_validation, predictor_labels, domain_labels = data

        predictions = self.model.predict(X_validation, batch_size=self.batch_size, verbose=0)
        label_predictions = predictions["label_predictor"]
        domain_predictions = np.round(predictions["domain_classifier"])
        
        ape_score = APE_score(predictor_labels, label_predictions).numpy()
        cosine_sim = cosine_similarity(predictor_labels, label_predictions)
        cosine_score = -tf.reduce_mean(cosine_sim).numpy()
        loss = self.loss["label_predictor"](predictor_labels, label_predictions).numpy()
        domain_accuracy = accuracy_score(domain_labels, domain_predictions)

        return {
            "ape_score": ape_score,
            "cosine_similarity": cosine_score,
            "loss": loss,
            "domain_accuracy": domain_accuracy,
        }
