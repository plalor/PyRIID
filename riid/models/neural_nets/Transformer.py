import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2

from riid import SampleSet, SpectraType, SpectraState, read_hdf
from riid.models.base import ModelInput, PyRIIDModel
from time import perf_counter as time


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
        if self.optimizer is None:
            self.optimizer = Adam(learning_rate=0.001)
        if self.metrics is None:
            self.metrics = [CategoricalCrossentropy()]
        if self.activity_regularizer is None:
            self.activity_regularizer = l1(0.0)
        if self.final_activation is None:
            self.final_activation = "softmax"

        self.model = None

    def fit(self, training_ss: SampleSet, validation_ss: SampleSet, batch_size: int = 200,
            epochs: int = 20, callbacks = None, patience: int = 10**4, es_monitor: str = "val_categorical_crossentropy",
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
            seq_length = input_shape // self.patch_size
            x = layers.Reshape((seq_length, self.patch_size), name="reshape_patches")(inputs)
            x = Dense(self.embed_dim, name="dense_embedding")(x)

            positions = tf.range(start=0, limit=seq_length, delta=1, name="position_indices")
            position_embedding = layers.Embedding(input_dim=seq_length, output_dim=self.embed_dim, name="position_embedding")(positions)
            x = x + position_embedding

            for layer in range(self.num_layers):
                attention_output = layers.MultiHeadAttention(num_heads=self.num_heads, 
                                                key_dim=self.embed_dim // self.num_heads,
                                                name=f"multi_head_attention_{layer}")(x, x)
                if self.dropout > 0:
                    attention_output = Dropout(self.dropout, name=f"mha_dropout_{layer}")(attention_output)
                attention_output = layers.LayerNormalization(epsilon=1e-6, name=f"mha_layernorm_{layer}")(x + attention_output)
                ffn_output = layers.Dense(self.ff_dim, activation=self.activation, name=f"ffn_dense1_{layer}")(attention_output)
                ffn_output = layers.Dense(self.embed_dim, name=f"ffn_dense2_{layer}")(ffn_output)
                if self.dropout > 0:
                    ffn_output = Dropout(self.dropout, name=f"ffn_dropout_{layer}")(ffn_output)
                x = layers.LayerNormalization(epsilon=1e-6, name=f"ffn_layernorm_{layer}")(attention_output + ffn_output)
            
            x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
            if self.dropout > 0:
                x = Dropout(self.dropout, name="dropout_layer")(x)

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

        t0 = time()
        history = self.model.fit(
            training_dataset,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_dataset,
            callbacks=callbacks,
         )
        self.history = history.history
        self.history["training_time"] = time() - t0

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs=model_outputs,
            normalization=training_ss.spectra_state,
        )

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
