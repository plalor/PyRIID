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


class DAMLP(PyRIIDModel):
    """Domain Adversarial Multi-layer Perceptron classifier."""
    def __init__(self, activation=None, loss=None, optimizer=None,
                 metrics=None, l2_alpha: float = 1e-4,
                 activity_regularizer=None, final_activation=None,
                 hidden_layers=None, grl_layer_size = 0,
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
            hidden_layers: hidden layer structure of the MLP
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
        self.grl_layer_size = grl_layer_size
        self.gamma = gamma
        self.dropout = dropout

        if self.activation is None:
            self.activation = "relu"
        if self.loss is None:
            self.loss = CategoricalCrossentropy()
        if self.optimizer is None:
            self.optimizer = Adam(learning_rate=0.001)
        if self.activity_regularizer is None:
            self.activity_regularizer = l1(0.0)
        if self.final_activation is None:
            self.final_activation = "softmax"

        self.model = None

    def fit(self, source_training_ss: SampleSet, target_training_ss: SampleSet, batch_size: int = 200, 
            epochs: int = 20, target_level="Isotope", verbose: bool = False):
        """Fit a model to the given `SampleSet`(s).

        Args:
            source_training_ss: `SampleSet` of `n` training spectra from the source domain where `n` >= 1 
                and the spectra are either foreground (AKA, "net") or gross.
            target_training_ss: `SampleSet` of `m` training spectra from the target domain where `m` >= 1
                and the spectra are either foreground (AKA, "net") or gross.
            batch_size: number of samples per gradient update
            epochs: maximum number of training iterations
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

        if not self.model:
            input_shape = X_train.shape[1]
            inputs = Input(shape=(input_shape,), name="Spectrum")
            if self.hidden_layers is None:
                self.hidden_layers = (input_shape//2,)
            x = inputs
            for layer, nodes in enumerate(self.hidden_layers):
                x = Dense(
                    nodes,
                    activation=self.activation,
                    activity_regularizer=self.activity_regularizer,
                    kernel_regularizer=l2(self.l2_alpha),
                    name=f"dense_{layer}"
                )(x)

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

        lambda_scheduler = LambdaScheduler(gamma=self.gamma, total_epochs=epochs, grl_layer=grl_layer)
        
        history = self.model.fit(
            x=X_train,
            y=labels_dict_train,
            sample_weight=sample_weight_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=(lambda_scheduler),
            shuffle=False,
        )
        self.history = history.history

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs=model_outputs,
            normalization=source_training_ss.spectra_state,
        )

        return history

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

        results = self.model.predict(X, batch_size=1000)["label_predictor"]

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

    def calc_domain_accuracy(self, ss: SampleSet, domain_label: int):
        """Calculate the domain classification accuracy on ss"""
        x_test = ss.get_samples().astype(float)
        preds = self.model.predict(x_test, batch_size=1000)
        domain_preds = np.round(preds["domain_classifier"]).astype(int).flatten()
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
