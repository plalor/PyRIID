import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, cosine_similarity
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2

from riid import SampleSet, SpectraType, SpectraState, read_hdf
from riid.models.base import ModelInput, PyRIIDModel
from riid.metrics import APE_score
from time import perf_counter as time


class ADDA(PyRIIDModel):
    """Adversarial Discriminative Domain Adaptation classifier"""            
    def __init__(self, activation=None, d_optimizer=None,
                 t_optimizer=None, metrics=None, source_model=None,
                 dense_layer_size=0):
        """
        Args:
            activation: activation function to use for discriminator dense layer
            d_optimizer: tensorflow optimizer or optimizer name to use for training discriminator
            t_optimizer: tensorflow optimizer or optimizer name to use for training target encoder
            metrics: list of metrics to be evaluating during training
            source_model: pretrained source model
            dense_layer_size: size of the dense layer in the discriminator
        """
        super().__init__()

        self.activation = activation
        self.d_optimizer = d_optimizer
        self.t_optimizer = t_optimizer
        self.metrics = metrics
        self.dense_layer_size = dense_layer_size

        ### Define models
        all_layers = source_model.layers
        encoder_input = source_model.input
        encoder_output = all_layers[-2].output
        self.source_encoder = Model(inputs=encoder_input, outputs=encoder_output, name="source_encoder")
        self.source_encoder.trainable = False
        
        classifier_input = Input(shape=encoder_output.shape[1:], name="encoder_output")
        classifier_output = all_layers[-1](classifier_input)
        self.source_classifier = Model(inputs=classifier_input, outputs=classifier_output, name="source_classifier")
        
        self.target_encoder = clone_model(self.source_encoder)
        self.target_encoder.build(self.source_encoder.input_shape)
        self.target_encoder.set_weights(self.source_encoder.get_weights())

        if self.activation is None:
            self.activation = "relu"
        if self.d_optimizer is None:
            self.d_optimizer = Adam(learning_rate=0.001)
        if self.t_optimizer is None:
            self.t_optimizer = Adam(learning_rate=0.001)
        if self.metrics is None:
            self.metrics = [APE_score]

        self.discriminator = None
        self.model = None
        self.loss = BinaryCrossentropy()

    def fit(self, source_training_ss: SampleSet, target_training_ss: SampleSet, source_validation_ss: SampleSet, 
            target_validation_ss: SampleSet, batch_size: int = 200, epochs: int = 20, callbacks = None, 
            target_level="Isotope", verbose: bool = False):
        """Fit a model to the given `SampleSet`(s).

        Args:
            training_ss: `SampleSet` of `n` training spectra where `n` >= 1 and the spectra 
                are either foreground (AKA, "net") or gross.
            validation_ss: `SampleSet` of `n` validation spectra where `n` >= 1 and the spectra 
                are either foreground (AKA, "net") or gross.
            batch_size: number of samples per gradient update
            epochs: maximum number of training iterations
            callbacks: list of callbacks to be passed to the TensorFlow `Model.fit()` method
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
        X_source_validation = source_validation_ss.get_samples()
        X_target_validation = target_validation_ss.get_samples()

        # Domain labels: 0 for source, 1 for target
        Y_source_train = np.zeros(len(X_source_train)).reshape(-1, 1)
        Y_target_train = np.ones(len(X_target_train)).reshape(-1, 1)
        Y_source_validation = np.zeros(len(X_source_validation)).reshape(-1, 1)
        Y_target_validation = np.ones(len(X_target_validation)).reshape(-1, 1)

        # Convert to tensors
        X_source_train = tf.convert_to_tensor(X_source_train, dtype=tf.float32)
        X_target_train = tf.convert_to_tensor(X_target_train, dtype=tf.float32)
        X_source_validation = tf.convert_to_tensor(X_source_validation, dtype=tf.float32)
        X_target_validation = tf.convert_to_tensor(X_target_validation, dtype=tf.float32)
        Y_source_train = tf.convert_to_tensor(Y_source_train, dtype=tf.float32)
        Y_target_train = tf.convert_to_tensor(Y_target_train, dtype=tf.float32)
        Y_source_validation = tf.convert_to_tensor(Y_source_validation, dtype=tf.float32)
        Y_target_validation = tf.convert_to_tensor(Y_target_validation, dtype=tf.float32)

        # Make datasets
        source_dataset_train = tf.data.Dataset.from_tensor_slices((X_source_train, Y_source_train))
        source_dataset_train = source_dataset_train.shuffle(buffer_size=len(X_source_train)).batch(batch_size)
        source_dataset_validation = tf.data.Dataset.from_tensor_slices((X_source_validation, Y_source_validation))
        source_dataset_validation = source_dataset_validation.shuffle(buffer_size=len(X_source_validation)).batch(batch_size)
        target_dataset_train = tf.data.Dataset.from_tensor_slices((X_target_train, Y_target_train))
        target_dataset_train = target_dataset_train.shuffle(buffer_size=len(X_target_train)).batch(batch_size)
        target_dataset_validation = tf.data.Dataset.from_tensor_slices((X_target_validation, Y_target_validation))
        target_dataset_validation = target_dataset_validation.shuffle(buffer_size=len(X_target_validation)).batch(batch_size)
                  
        ### Build discriminator
        if not self.discriminator:
            input_shape = self.source_encoder.output_shape[1]
            inputs = Input(shape=(input_shape,), name="features")
            x = inputs
            if self.dense_layer_size > 0:
                x = Dense(self.dense_layer_size, activation=self.activation)(x)
            output = Dense(1, activation="sigmoid", name="discriminator")(x)
            self.discriminator = Model(inputs, output)

        # Define ADDA model using target encoder and source classifier
        self.model = Model(
            inputs=self.target_encoder.input,
            outputs=self.source_classifier(self.target_encoder.output)
        )
        self.model.compile(loss = CategoricalCrossentropy())
        self._update_info(
            target_level=target_level,
            model_outputs = source_training_ss.sources.T.groupby(target_level, sort=False).sum().T.columns.values.tolist(),
            normalization=source_training_ss.spectra_state,
        )
        self._set_predict_fn()

        # Training loop
        self.history = {'d_loss': [], 't_loss': [], 'train_ape_score': [], 'val_ape_score': []}
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}...", end="")
                t0 = time()
    
            for (x_s, y_s), (x_t, y_t) in zip(source_dataset_train, target_dataset_train):
                d_loss = self.train_discriminator_step(x_s, x_t, y_s, y_t)
                t_loss = self.train_target_encoder_step(x_t)
    
            self.history['d_loss'].append(float(d_loss))
            self.history['t_loss'].append(float(t_loss))

            train_ape_score = self.calc_APE_score(target_training_ss)
            val_ape_score = self.calc_APE_score(target_validation_ss)
            
            self.history[f"train_ape_score"].append(train_ape_score)
            self.history[f"val_ape_score"].append(val_ape_score)

            if verbose:
                print(f"finished in {time()-t0:.0f} seconds")
                print(f"  d_loss={d_loss:.4f}  t_loss={t_loss:.4f}  "
                      f"train_ape_score={train_ape_score:.4f}  "
                      f"val_ape_score={val_ape_score:.4f}")
                
        return self.history

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
        """Calculate the loss on ss"""
        self.predict(ss)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        loss = self.model.loss(y_true, y_pred).numpy()
        return loss

    @tf.function
    def train_discriminator_step(self, x_s, x_t, y_s, y_t):
        """
        1. Freeze target_encoder
        2. Update discriminator on source+target features
        """
        self.target_encoder.trainable = False
        self.discriminator.trainable = True
    
        with tf.GradientTape() as tape:
            f_s = self.source_encoder(x_s, training=False)
            f_t = self.target_encoder(x_t, training=False)
    
            pred_s = self.discriminator(f_s, training=True)
            pred_t = self.discriminator(f_t, training=True)
    
            loss_s = self.loss(y_s, pred_s)
            loss_t = self.loss(y_t, pred_t)
            d_loss = (loss_s + loss_t) / 2.0
    
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
    
        return d_loss
    
    
    @tf.function
    def train_target_encoder_step(self, x_t):
        """
        1. Freeze discriminator
        2. Update target_encoder so that D thinks target features are 'source'
        """
        self.target_encoder.trainable = True
        self.discriminator.trainable = False
    
        with tf.GradientTape() as tape:
            f_t = self.target_encoder(x_t, training=True)
            pred_t = self.discriminator(f_t, training=False)
            target_labels = tf.zeros_like(pred_t)
            t_loss = self.loss(target_labels, pred_t)
    
        grads = tape.gradient(t_loss, self.target_encoder.trainable_variables)
        self.t_optimizer.apply_gradients(zip(grads, self.target_encoder.trainable_variables))
    
        return t_loss