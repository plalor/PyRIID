import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout, Lambda, Embedding, Add, \
    MultiHeadAttention, LayerNormalization, Layer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel
from time import perf_counter as time


class Transformer(PyRIIDModel):
    """Transformer classifier."""
    def __init__(self, activation=None, loss=None, optimizer=None, metrics=None,
                 final_activation=None, embed_dim=None, num_heads=None, ff_dim=None,
                 num_layers=None, patch_size=None, stride=None, dropout=0):
        """
        Args:
            activation: activation function to use for each dense layer
            loss: loss function to use for training
            optimizer: tensorflow optimizer or optimizer name to use for training
            metrics: list of metrics to be evaluating during training
            final_activation: final activation function to apply to model output
            embed_dim: size of the embedding vector
            num_heads: number of attention heads
            ff_dim: dimension of feed-forward network
            num_layers: number of transformer blocks
            patch_size: size of patches to reshape the input spectrum
            stride: step size between each patch
            dropout: optional droupout layer after each hidden layer
        """
        super().__init__()

        self.activation = activation or "relu"
        self.loss = loss or CategoricalCrossentropy()
        self.optimizer = optimizer or Adam(learning_rate=0.001)
        self.metrics = metrics
        self.final_activation = final_activation or "softmax"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.stride = stride or self.patch_size
        self.dropout = dropout

        self.model = None

    def fit(self, training_ss: SampleSet, validation_ss: SampleSet, batch_size: int = 200,
            epochs: int = 20, callbacks = None, patience: int = 10**4, es_monitor: str = "val_loss",
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

            num_patches = (input_shape - self.patch_size) // self.stride + 1

            x = Lambda(
                extract_patches,
                arguments={"patch_size": self.patch_size, "stride": self.stride},
                name="extract_patches"
            )(inputs)
            
            x = Dense(self.embed_dim, name="dense_embedding")(x)

            pos_indices = Lambda(
                make_positions,
                arguments={"num_patches": num_patches},
                name="make_positions"
            )(x)

            pos_embed = Embedding(
                input_dim=num_patches,
                output_dim=self.embed_dim,
                name="position_embedding"
            )(pos_indices)

            x = Add(name="add_pos")([x, pos_embed])
            if self.dropout > 0:
                x = Dropout(self.dropout)(x)

            x = ClassToken(self.embed_dim, name="class_token")(x)

            for layer in range(self.num_layers):
                y = LayerNormalization(epsilon=1e-6, name=f"pre_mha_layernorm_{layer}")(x)
                y = MultiHeadAttention(num_heads=self.num_heads,
                        key_dim=self.embed_dim // self.num_heads,
                        name=f"multi_head_attention_{layer}")(y, y)
                if self.dropout > 0:
                    y = Dropout(self.dropout, name=f"mha_dropout_{layer}")(y)
                x = x + y

                y = LayerNormalization(epsilon=1e-6, name=f"pre_ffn_layernorm_{layer}")(x)
                y = Dense(self.ff_dim, activation=self.activation, name=f"ffn_dense1_{layer}")(y)
                if self.dropout > 0:
                    y = Dropout(self.dropout, name=f"ffn_dropout1_{layer}")(y)
                y = Dense(self.embed_dim, name=f"ffn_dense2_{layer}")(y)
                if self.dropout > 0:
                    y = Dropout(self.dropout, name=f"ffn_dropout2_{layer}")(y)
                x = x + y
            
            x = Lambda(lambda t: t[:, 0, :], name="take_cls_token")(x)  # shape (batch, embed_dim)
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

### Need to decorate with serialization API to save/load model
from tensorflow.keras.saving import register_keras_serializable

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
