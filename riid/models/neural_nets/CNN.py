import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten, Lambda, BatchNormalization, \
    Activation, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel
from riid.models.functions import zscore, sqrt_zscore, add_channel
from riid.models.callbacks import TimeLimitCallback


class CNN(PyRIIDModel):
    """Convolutional neural network classifier."""
    def __init__(self, activation="relu", loss=None, optimizer=None,
                 metrics=None, final_activation="softmax", filters=None,
                 kernel_sizes=None, dense_layer_sizes=None, readout="flatten",
                 dropout=0, normalize="zscore"):
        """
        Args:
            activation: activation function to use for each dense layer
            loss: loss function to use for training
            optimizer: tensorflow optimizer or optimizer name to use for training
            metrics: list of metrics to be evaluating during training
            final_activation: final activation function to apply to model output
            filters: list of number of filters for each conv layer of the CNN
            kernel_sizes: list of kernel sizes for each conv layer of the CNN
            dense_layer_sizes: sizes of the final dense layers after the convolutional layers
            readout: strategy for aggregating token embeddings into a vector for classification
            dropout: optional dropout layer after each hidden layer
            normalize: whether (and how) to normalize input spectra
        """
        super().__init__()

        self.activation = activation
        self.loss = loss or CategoricalCrossentropy()
        self.optimizer = optimizer or Adam(learning_rate=0.001)
        self.metrics = metrics
        self.final_activation = final_activation

        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dense_layer_sizes = dense_layer_sizes
        self.readout = readout
        self.dropout = dropout
        self.normalize = normalize

        self.model = None

    def fit(self, training_ss: SampleSet, validation_ss: SampleSet, batch_size=64, epochs=None,
            callbacks=None, patience=10**9, es_monitor="val_loss", es_mode="min", es_verbose=0,
            target_level="Isotope", verbose=False, training_time=None):
        """Fit a model to the given `SampleSet`(s).

        Args:
            training_ss: `SampleSet` of `n` training spectra where `n` >= 1 and the spectra 
                are either foreground (AKA, "net") or gross.
            validation_ss: `SampleSet` of `n` validation spectra where `n` >= 1 and the spectra 
                are either foreground (AKA, "net") or gross.
            batch_size: number of samples per gradient update
            epochs: maximum number of training iterations
            callbacks: list of callbacks to be passed to the TensorFlow `Model.fit()` method
            patience: number of epochs to wait for `EarlyStopping` object
            es_monitor: quantity to be monitored for `EarlyStopping` object
            es_mode: mode for `EarlyStopping` object
            es_verbose: verbosity level for `EarlyStopping` object
            target_level: `SampleSet.sources` column level to use
            verbose: whether to show detailed model training output
            training_time: whether to terminate early if run exceeds prealloted time

        Returns:
            `history` dictionary

        Raises:
            `ValueError` when no spectra are provided as input
        """

        if training_ss.n_samples <= 0:
            raise ValueError("No spectr[a|um] provided!")

        if training_ss.spectra_type == SpectraType.Gross:
            self.model_inputs = (ModelInput.GrossSpectrum,)
        elif training_ss.spectra_type == SpectraType.Foreground:
            self.model_inputs = (ModelInput.ForegroundSpectrum,)
        elif training_ss.spectra_type == SpectraType.Background:
            self.model_inputs = (ModelInput.BackgroundSpectrum,)
        else:
            raise ValueError(f"{training_ss.spectra_type} is not supported in this model.")
                
        X_train = training_ss.get_samples().astype("float32")
        source_contributions_df_train = training_ss.sources.T.groupby(target_level, sort=False).sum().T
        model_outputs = source_contributions_df_train.columns.values.tolist()
        Y_train = source_contributions_df_train.values.astype("float32")
        
        X_validation = validation_ss.get_samples().astype("float32")
        Y_validation = validation_ss.sources.T.groupby(target_level, sort=False).sum().T.values.astype("float32")

        training_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        training_dataset = training_dataset.shuffle(len(X_train)).batch(batch_size)
        validation_dataset = tf.data.Dataset.from_tensor_slices((X_validation, Y_validation))
        validation_dataset = validation_dataset.batch(batch_size)

        if not self.model:
            input_shape = X_train.shape[1]
            inputs = Input(shape=(input_shape,), name="Spectrum")
            if self.normalize == "zscore":
                x = Lambda(zscore, name="zscore")(inputs)
            elif self.normalize == "sqrt_zscore":
                x = Lambda(sqrt_zscore, name="sqrt_zscore")(inputs)
            else:
                x = inputs

            x = Lambda(add_channel, name="add_channel")(x)
            for layer, (filters, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes)):
                x = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    activation=None,
                    use_bias=False,
                    name=f"conv_{layer}"
                )(x)
                x = BatchNormalization(name=f"batchnorm_{layer}")(x)
                x = Activation(self.activation, name=f"activation_{layer}")(x)
                x = MaxPooling1D(pool_size=2, name=f"maxpool_{layer}")(x)

            if self.readout == "flatten":
                x = Flatten(name="flatten")(x)
            elif self.readout == "gap":
                x = GlobalAveragePooling1D(name="gap")(x)
            elif self.readout == "gmp":
                x = GlobalMaxPooling1D(name="gmp")(x)
            elif self.readout == "gap_gmp":
                x = Concatenate(name="gap_gmp")([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])
            else:
                raise ValueError("`self.readout` must be 'gap','flatten','gmp','gap_gmp'")

            for layer, nodes in enumerate(self.dense_layer_sizes):
                x = Dense(nodes, activation=self.activation, name=f"dense_{layer}")(x)
                x = Dropout(self.dropout, name=f"dense_dropout_{layer}")(x)
            outputs = Dense(Y_train.shape[1], activation=self.final_activation, name="output")(x)
            self.model = Model(inputs, outputs)
            self.model.compile(loss=self.loss, optimizer=self.optimizer,
                               metrics=self.metrics)

        es = EarlyStopping(
            monitor=es_monitor,
            patience=patience,
            verbose=es_verbose,
            restore_best_weights=True,
            mode=es_mode,
        )
        if callbacks:
            callbacks.append(es)
        else:
            callbacks = [es]

        if training_time is not None:
            callbacks.append(TimeLimitCallback(training_time))
            if epochs is None:
                epochs = 10**9

        # For small datasets, we shouldn't perform a validation callback every epoch
        validation_freq = max(1, int((len(X_validation) / len(X_train))**0.3))

        history = self.model.fit(
            training_dataset,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_dataset,
            validation_freq=validation_freq,
            callbacks=callbacks,
         )
        self.history = history.history

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs=model_outputs,
            normalization=training_ss.spectra_state,
        )

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
