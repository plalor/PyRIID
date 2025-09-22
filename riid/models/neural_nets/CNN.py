import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Dense, Input, Dropout, SpatialDropout1D, Conv1D, MaxPooling1D, Flatten, Lambda
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel
from time import perf_counter as timer


class CNN(PyRIIDModel):
    """Convolutional neural network classifier."""
    def __init__(self, activation="relu", loss=None, optimizer=None,
                 metrics=None, l2_alpha=None, activity_regularizer=None,
                 final_activation="softmax", convolutional_layers=None,
                 dense_layer_sizes=None, dropout=0):
        """
        Args:
            activation: activation function to use for each dense layer
            loss: loss function to use for training
            optimizer: tensorflow optimizer or optimizer name to use for training
            metrics: list of metrics to be evaluating during training
            l2_alpha: alpha value for the L2 regularization of each dense layer
            activity_regularizer: regularizer function applied each dense layer output
            final_activation: final activation function to apply to model output
            convolutional_layers: (filter, kernel_size) of each conv layer of the CNN
            dense_layer_sizes: sizes of the final dense layers after the convolutional layers
            dropout: optional droupout layer after each hidden layer
        """
        super().__init__()

        self.activation = activation
        self.loss = loss or CategoricalCrossentropy()
        self.optimizer = optimizer or Adam(learning_rate=0.001)
        self.metrics = metrics
        self.kernel_regularizer = l2(l2_alpha) if l2_alpha else None
        self.activity_regularizer = activity_regularizer
        self.final_activation = final_activation

        self.convolutional_layers = convolutional_layers
        self.dense_layer_sizes = dense_layer_sizes
        self.dropout = dropout

        self.model = None

    def fit(self, training_ss: SampleSet, validation_ss: SampleSet, batch_size=64, epochs=None,
            callbacks=None, patience=10**9, es_monitor="val_loss", es_mode="min", es_verbose=0,
            target_level="Isotope", verbose=False, training_time=None, normalize=True):
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
            normalize: whether to apply z-score normalization to input spectra

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
            x = Lambda(zscore, name="zscore")(inputs) if normalize else inputs
            x = Lambda(add_channel, name="add_channel")(x)
            for layer, (filters, kernel_size) in enumerate(self.convolutional_layers):
                x = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    activation=self.activation,
                    activity_regularizer=self.activity_regularizer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"conv_{layer}"
                )(x)
                x = MaxPooling1D(pool_size=2, name=f"maxpool_{layer}")(x)
                x = SpatialDropout1D(self.dropout, name=f"conv_dropout_{layer}")(x)

            x = Flatten(name="flatten")(x)
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

class TimeLimitCallback(Callback):
    def __init__(self, max_seconds):
        super().__init__()
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = timer()

    def on_epoch_end(self, epoch, logs=None):
        if timer() - self.start_time >= self.max_seconds:
            self.model.stop_training = True

### Need to decorate with serialization API to save/load model
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom", name="add_channel")
def add_channel(inputs):
    return tf.expand_dims(inputs, -1)

@register_keras_serializable(package="Custom", name="zscore")
def zscore(x):
    m = tf.reduce_mean(x, axis=-1, keepdims=True)
    s = tf.math.reduce_std(x, axis=-1, keepdims=True)
    return (x - m) / s
