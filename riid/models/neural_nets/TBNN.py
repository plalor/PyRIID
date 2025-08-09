import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Dense, Input, Dropout, Lambda, Embedding, Add, MultiHeadAttention, \
    LayerNormalization, Layer, Conv1D, SpatialDropout1D, TimeDistributed, MaxPooling1D, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel
from time import perf_counter as timer


class TBNN(PyRIIDModel):
    """Transformer-based neural network classifier."""
    def __init__(self, activation="relu", loss=None, optimizer=None, metrics=None,
                 final_activation="softmax", embed_mode="linear", embed_filters=None,
                 embed_dim=None, pos_encoding="learnable", num_heads=None, ff_dim=None,
                 num_layers=None, patch_size=None, stride=None, dropout=0):
        """
        Args:
            activation: activation function to use for each dense layer
            loss: loss function to use for training
            optimizer: tensorflow optimizer or optimizer name to use for training
            metrics: list of metrics to be evaluating during training
            final_activation: final activation function to apply to model output
            embed_mode: mode for performing the embedding
            embed_filters: number of filters for intermediate layers in CNN and MLP embeddings
            embed_dim: size of the embedding vector
            pos_encoding: whether to use `learnable` or `sinusoidal` positional encodings
            num_heads: number of attention heads
            ff_dim: dimension of feed-forward network
            num_layers: number of transformer blocks
            patch_size: size of patches to reshape the input spectrum
            stride: step size between each patch
            dropout: optional droupout layer after each hidden layer
        """
        super().__init__()

        self.activation = activation
        self.loss = loss or CategoricalCrossentropy()
        self.optimizer = optimizer or Adam(learning_rate=0.001)
        self.metrics = metrics
        self.final_activation = final_activation

        self.embed_mode = embed_mode.lower()
        self.embed_filters = embed_filters
        self.embed_dim = embed_dim or patch_size
        self.pos_encoding = pos_encoding.lower()
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.dropout = dropout

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

            if self.embed_mode == "raw":
                assert self.embed_dim == self.patch_size
                inputs = Input(shape=(input_shape,), name="Spectrum")
                x = Lambda(
                    extract_patches,
                    arguments={"patch_size": self.patch_size, "stride": self.stride},
                    name="patch_projection"
                )(inputs)

            elif self.embed_mode == "linear":
                inputs = Input(shape=(input_shape,1), name="Spectrum")
                x = Conv1D(
                    filters=self.embed_dim,
                    kernel_size=self.patch_size,
                    strides=self.stride,
                    padding="valid",
                    use_bias=True,
                    name="patch_projection"
                )(inputs)

            elif self.embed_mode == "mlp_single":
                inputs = Input(shape=(input_shape,1), name="Spectrum")
                x = Conv1D(
                    filters=self.embed_dim,
                    kernel_size=self.patch_size,
                    strides=self.stride,
                    padding="valid",
                    use_bias=True,
                    activation=self.activation,
                    name="patch_projection"
                )(inputs)

            elif self.embed_mode == "mlp_double":
                self.embed_filters = self.embed_filters or self.embed_dim
                inputs = Input(shape=(input_shape,1), name="Spectrum")
                x = Conv1D(
                  filters=self.embed_filters,
                  kernel_size=self.patch_size,
                  strides=self.stride,
                  padding="valid",
                  activation=self.activation,
                  name="patch_projection1"
                )(inputs)
                x = SpatialDropout1D(self.dropout, name="drop_patch_projection1")(x)
                x = Conv1D(
                  filters=self.embed_dim,
                  kernel_size=1,
                  activation=self.activation,
                  name="patch_projection2"
                )(x)
                x = SpatialDropout1D(self.dropout, name="drop_patch_projection2")(x)

            elif self.embed_mode == "cnn_single":
                self.embed_filters = self.embed_filters or 16
                inputs = Input(shape=(input_shape,), name="Spectrum")
                patches = Lambda(
                    extract_patches,
                    arguments={"patch_size": self.patch_size, "stride": self.stride},
                    name="patch_extraction"
                )(inputs)
                patches = Lambda(add_channel, name="add_channel")(patches)
                def make_patch_cnn_single():
                    return Sequential([
                        Conv1D(self.embed_filters, kernel_size=3, padding="same", activation=self.activation),
                        SpatialDropout1D(self.dropout),
                        MaxPooling1D(pool_size=2),
                        Flatten(),
                        Dense(self.embed_dim, activation=self.activation),
                        Dropout(self.dropout),
                    ])
                x = TimeDistributed(make_patch_cnn_single(), name="patch_cnn_single")(patches)

            elif self.embed_mode == "cnn_double":
                self.embed_filters = self.embed_filters or 16
                inputs = Input(shape=(input_shape,), name="Spectrum")
                patches = Lambda(
                    extract_patches,
                    arguments={"patch_size": self.patch_size, "stride": self.stride},
                    name="patch_extraction"
                )(inputs)
                patches = Lambda(add_channel, name="add_channel")(patches)
                def make_patch_cnn_double():
                    return Sequential([
                        Conv1D(self.embed_filters, kernel_size=3, padding="same", activation=self.activation),
                        SpatialDropout1D(self.dropout),
                        MaxPooling1D(pool_size=2),
                        Conv1D(2 * self.embed_filters, kernel_size=3, padding="same", activation=self.activation),
                        SpatialDropout1D(self.dropout),
                        MaxPooling1D(pool_size=2),
                        Flatten(),
                        Dense(self.embed_dim, activation=self.activation),
                        Dropout(self.dropout),
                    ])
                x = TimeDistributed(make_patch_cnn_double(), name="patch_cnn_double")(patches)

            else:
                raise ValueError("`embed_mode` not understood.")
            
            num_patches = (input_shape - self.patch_size) // self.stride + 1
            x = SpatialDropout1D(self.dropout, name="drop_embed")(x)

            x = ClassToken(self.embed_dim, name="class_token")(x)
            num_tokens = num_patches + 1

            if self.pos_encoding == "learnable":
                pos_indices = Lambda(
                    make_positions,
                    arguments={"num_patches": num_tokens},
                    name="make_positions"
                )(x)
                pos_embed = Embedding(
                    input_dim=num_tokens,
                    output_dim=self.embed_dim,
                    name="position_embedding"
                )(pos_indices)

            elif self.pos_encoding == "sinusoidal":
                pos_embed = Lambda(
                    add_sinusoidal_pos,
                    arguments={"num_patches": num_tokens, "embed_dim": self.embed_dim},
                    output_shape=(num_tokens, self.embed_dim),
                    name="sinusoidal_pos"
                )(x)
                
            else:
                raise ValueError("`pos_encoding` must be 'learnable' or 'sinusoidal'")

            x = Add(name="add_pos")([x, pos_embed])
            x = Dropout(self.dropout, name="drop_tokens")(x)

            for layer in range(self.num_layers):
                y = LayerNormalization(epsilon=1e-6, name=f"ln_attn_{layer}")(x)
                y = MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.embed_dim // self.num_heads,
                    dropout=self.dropout,
                    name=f"mha_{layer}"
                )(y, y)
                y = Dropout(self.dropout, name=f"drop_attn_{layer}")(y)
                x = Add(name=f"resid_attn_{layer}")([x, y])

                y = LayerNormalization(epsilon=1e-6, name=f"ln_ffn_{layer}")(x)
                y = Dense(self.ff_dim, activation=self.activation, name=f"ffn1_{layer}")(y)
                y = Dropout(self.dropout, name=f"drop_ffn_in_{layer}")(y)
                y = Dense(self.embed_dim, name=f"ffn2_{layer}")(y)
                y = Dropout(self.dropout, name=f"drop_ffn_out_{layer}")(y)
                x = Add(name=f"resid_ffn_{layer}")([x, y])

            x = LayerNormalization(epsilon=1e-6, name="encoder_norm")(x)
            x = Lambda(take_cls_token_fn, name="take_cls_token")(x)
            x = Dropout(self.dropout, name="dropout_out")(x)

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
        validation_freq = max(1, int(np.sqrt(len(X_validation) / len(X_train))))

        t0 = timer()
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

@register_keras_serializable(package="Custom", name="ClassToken")
class ClassToken(Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # input_shape = (batch, num_patches, embed_dim)
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token"
        )
        super().build(input_shape)

    def call(self, x):
        # x.shape = (batch, num_patches, embed_dim)
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_tokens, x], axis=1)

@register_keras_serializable(package="Custom", name="take_cls_token_fn")
def take_cls_token_fn(x):
    return x[:, 0, :]

@register_keras_serializable(package="Custom", name="extract_patches")
def extract_patches(x, patch_size, stride):
    return tf.signal.frame(
        x,
        frame_length=patch_size,
        frame_step=stride,
        axis=1
    )

@register_keras_serializable(package="Custom", name="make_positions")
def make_positions(x, num_patches):
    batch = tf.shape(x)[0]
    idx = tf.range(num_patches)[tf.newaxis, :]
    return tf.tile(idx, [batch, 1])

@register_keras_serializable(package="Custom", name="add_sinusoidal_pos")
def add_sinusoidal_pos(x, num_patches, embed_dim):
    pos = tf.cast(tf.range(num_patches)[:, None], tf.float32)
    i = tf.cast(tf.range(embed_dim)[None, :], tf.float32)
    angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(i/2.0)) / tf.cast(embed_dim, tf.float32))
    angle_rads = pos * angle_rates
    sin = tf.sin(angle_rads)
    cos = tf.cos(angle_rads)
    even_mask = tf.cast(tf.equal(tf.math.mod(tf.range(embed_dim), 2), 0), tf.float32)[None, :]
    pos_encoding = sin * even_mask + cos * (1.0 - even_mask)
    return tf.tile(pos_encoding[None, :, :], [tf.shape(x)[0], 1, 1])

@register_keras_serializable(package="Custom", name="add_channel")
def add_channel(inputs):
    """Function to add a channel dimension (serializable)."""
    # inputs shape: (batch, num_patches, patch_size)
    return tf.expand_dims(inputs, -1)  # -> (batch, num_patches, patch_size, 1)
