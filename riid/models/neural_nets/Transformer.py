# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains a simple transformer."""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import Sequence
from keras.api.callbacks import EarlyStopping
from keras.api import layers
from keras.api.layers import Dense, Input, Dropout, Flatten
from keras.api.losses import CategoricalCrossentropy, MeanAbsoluteError
from keras.api.models import Model
from keras.api.optimizers import Adam
from keras.api.regularizers import l1, l2
from sklearn.preprocessing import StandardScaler

from riid import SampleSet, SpectraType, SpectraState, read_hdf
from riid.models.base import ModelInput, PyRIIDModel
from riid.metrics import APE_score, cosine_similarity


class Transformer(PyRIIDModel):
    """Transformer classifier."""
    def __init__(self, activation=None, loss=None, optimizer=None,
                 metrics=None, l2_alpha: float = 1e-4,
                 activity_regularizer=None, final_activation=None,
                 embed_dim=None, num_heads=None, ff_dim=None, num_layers=None, 
                 patch_size=None, dropout=0):
        """
        Args:
            activation: activation function to use for each dense layer
            loss: loss function to use for training
            optimizer: tensorflow optimizer or optimizer name to use for training
            metrics: list of metrics to be evaluating during training
            l2_alpha: alpha value for the L2 regularization of each dense layer
            activity_regularizer: regularizer function applied each dense layer output
            final_activation: final activation function to apply to model output
            dropout: optional droupout layer after each hidden layer
            embed_dim: size of the embedding vector
            num_heads: number of attention heads
            ff_dim: dimension of feed-forward network
            num_layers: number of transformer blocks
        """
        super().__init__()

        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.l2_alpha = l2_alpha
        self.activity_regularizer = activity_regularizer
        self.final_activation = final_activation

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.dropout = dropout

        if self.activation is None:
            self.activation = "relu"
        if self.loss is None:
            self.loss = CategoricalCrossentropy()
        if optimizer is None:
            self.optimizer = Adam(learning_rate=0.001)
        if self.metrics is None:
            self.metrics = [APE_score]
        if self.activity_regularizer is None:
            self.activity_regularizer = l1(0.0)
        if self.final_activation is None:
            self.final_activation = "softmax"

        self.model = None
        self._set_predict_fn()


    def fit(self, training_ss: SampleSet, validation_ss: SampleSet, batch_size: int = 200,
            epochs: int = 20, callbacks = None, patience: int = 10, es_monitor: str = "val_loss",
            es_mode: str = "min", es_verbose=0, target_level="Isotope", verbose: bool = False):
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

        Returns:
            `tf.History` object.

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
                
        X_train = training_ss.get_samples()
        source_contributions_df_train = training_ss.sources.T.groupby(target_level, sort=False).sum().T
        model_outputs = source_contributions_df_train.columns.values.tolist()
        Y_train = source_contributions_df_train.values

        spectra_tensor_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        labels_tensor_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        training_dataset = tf.data.Dataset.from_tensor_slices((spectra_tensor_train, labels_tensor_train))
        training_dataset = training_dataset.batch(batch_size=batch_size)

        X_validation = validation_ss.get_samples()
        source_contributions_df_validation = validation_ss.sources.T.groupby(target_level, sort=False).sum().T
        Y_validation = source_contributions_df_validation.values

        spectra_tensor_validation = tf.convert_to_tensor(X_validation, dtype=tf.float32)
        labels_tensor_validation = tf.convert_to_tensor(Y_validation, dtype=tf.float32)
        validation_dataset = tf.data.Dataset.from_tensor_slices((spectra_tensor_validation, labels_tensor_validation))
        validation_dataset = validation_dataset.batch(batch_size=batch_size)

        if not self.model:
            input_shape = X_train.shape[1]
            inputs = Input(shape=(input_shape,), name="Spectrum")
            seq_length = input_shape // self.patch_size
            x = layers.Reshape((seq_length, self.patch_size))(inputs)
            x = Dense(self.embed_dim)(x)

            positions = tf.range(start=0, limit=seq_length, delta=1)
            position_embedding = layers.Embedding(input_dim=seq_length, output_dim=self.embed_dim)(positions)
            x = x + position_embedding

            for layer in range(self.num_layers):
                attention_output = layers.MultiHeadAttention(num_heads=self.num_heads, 
                                                key_dim=self.embed_dim // self.num_heads,
                                                name=f"multi_head_attention_{layer}")(x, x)
                if self.dropout > 0:
                    attention_output = Dropout(self.dropout)(attention_output)
                attention_output = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
                ffn_output = layers.Dense(self.ff_dim, activation=self.activation, name=f"ffn_dense1_{layer}")(attention_output)
                ffn_output = layers.Dense(self.embed_dim, name=f"ffn_dense2_{layer}")(ffn_output)
                if self.dropout > 0:
                    attention_output = Dropout(self.dropout)(attention_output)
                x = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
            
            x = layers.GlobalAveragePooling1D()(x)
            if self.dropout > 0:
                x = Dropout(self.dropout)(x)

            outputs = Dense(Y_train.shape[1], activation=self.final_activation)(x)
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

        history = self.model.fit(
            training_dataset,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_dataset,
            callbacks=callbacks,
         )
        self.history = history.history

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs=model_outputs,
            normalization=training_ss.spectra_state,
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
        return self.model(input_tensor, training=False)

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
        """Calculated the prediction F1 score on ss"""
        self.predict(ss)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        ape = APE_score(y_true, y_pred).numpy()
        return ape

    def calc_cosine_similarity(self, ss: SampleSet, target_level="Isotope"):
        """Calculated the prediction total absolute error score on ss"""
        self.predict(ss)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        cosine_score = cosine_similarity(y_true, y_pred).numpy()
        return cosine_score

    def calc_loss(self, ss: SampleSet, target_level="Isotope"):
        """Calculated the prediction total absolute error score on ss"""
        self.predict(ss)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        loss = self.model.loss(y_true, y_pred).numpy()
        return loss
