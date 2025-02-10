import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import BinaryCrossentropy, cosine_similarity
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType, SpectraState, read_hdf
from riid.models.base import ModelInput, PyRIIDModel
from riid.metrics import APE_score
from time import perf_counter as time


class ADDA(PyRIIDModel):
    """Adversarial Discriminative Domain Adaptation classifier"""            
    def __init__(self, activation=None, d_optimizer=None, t_optimizer=None,
                 source_model=None, dense_layer_size=0):
        """
        Args:
            activation: activation function to use for discriminator dense layer
            d_optimizer: tensorflow optimizer or optimizer name to use for training discriminator
            t_optimizer: tensorflow optimizer or optimizer name to use for training target encoder
            source_model: pretrained source model
            dense_layer_size: size of the dense layer in the discriminator
        """
        super().__init__()

        self.activation = activation
        self.d_optimizer = d_optimizer
        self.t_optimizer = t_optimizer
        self.dense_layer_size = dense_layer_size
        self.discriminator_loss = BinaryCrossentropy()

        if source_model is not None:
            self.classification_loss = source_model.loss

            # Remove dropout layers for stability
            config = source_model.get_config()
            filtered_layers = [layer for layer in config["layers"] if layer["class_name"] != "Dropout"]
            config["layers"] = filtered_layers
            self.source_model = Model.from_config(config)
            self.source_model.set_weights(source_model.get_weights())

            all_layers = self.source_model.layers
            encoder_input = self.source_model.input
            encoder_output = all_layers[-2].output
            self.source_encoder = Model(inputs=encoder_input, outputs=encoder_output, name="source_encoder")
            self.source_encoder.trainable = False

            classifier_input = Input(shape=encoder_output.shape[1:], name="encoder_output")
            classifier_output = all_layers[-1](classifier_input)
            self.source_classifier = Model(inputs=classifier_input, outputs=classifier_output, name="source_classifier")

            self.target_encoder = clone_model(self.source_encoder)
            self.target_encoder._name = "target_encoder"
            self.target_encoder.build(self.source_encoder.input_shape)
            self.target_encoder.set_weights(self.source_encoder.get_weights())
        else:
            raise ValueError("A pretrained source model must be provided")
            
        if self.activation is None:
            self.activation = "relu"
        if self.d_optimizer is None:
            self.d_optimizer = Adam(learning_rate=0.001)
        if self.t_optimizer is None:
            self.t_optimizer = Adam(learning_rate=0.001)

        self.discriminator = None
        self.model = None

    def fit(self, source_ss: SampleSet, target_ss: SampleSet, validation_ss: SampleSet,
            batch_size: int = 200, epochs: int = 20, target_level="Isotope", verbose: bool = False):
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
        X_source = source_ss.get_samples()
        X_target = target_ss.get_samples()

        # Domain labels: 0 for source, 1 for target
        Y_source = np.zeros(len(X_source)).reshape(-1, 1)
        Y_target = np.ones(len(X_target)).reshape(-1, 1)

        # Make datasets
        source_dataset = tf.data.Dataset.from_tensor_slices((X_source, Y_source))
        source_dataset = source_dataset.shuffle(len(X_source)).batch(batch_size)
        target_dataset = tf.data.Dataset.from_tensor_slices((X_target, Y_target))
        target_dataset = target_dataset.shuffle(len(X_target)).batch(batch_size)
                  
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
        self.model.compile(loss = self.classification_loss)

        self._update_info(
            target_level=target_level,
            model_outputs = source_ss.sources.T.groupby(target_level, sort=False).sum().T.columns.values.tolist(),
            normalization=source_ss.spectra_state,
        )

        # Training loop
        self.history = {"d_loss": [], "t_loss": [], "val_ape_score": []}
        best_val_ape = -np.inf
        best_weights = None
        t0 = time()
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
                t1 = time()

            for (x_s, y_s), (x_t, y_t) in zip(source_dataset, target_dataset):
                d_loss = self.train_discriminator_step(x_s, x_t, y_s, y_t)
                t_loss = self.train_target_encoder_step(x_t)

            val_ape_score = self.calc_APE_score(validation_ss, target_level=target_level)
            self.history["d_loss"].append(float(d_loss))
            self.history["t_loss"].append(float(t_loss))
            self.history["val_ape_score"].append(val_ape_score)

            # Save best model weights based on the validation APE score
            if val_ape_score > best_val_ape:
                best_val_ape = val_ape_score
                best_weights = self.model.get_weights()

            if verbose:
                print(f"Finished in {time()-t1:.0f} seconds")
                print(f"  d_loss: {d_loss:.4f} - t_loss: {t_loss:.4f} - val_ape_score: {val_ape_score:.4f}")

        self.history["training_time"] = time() - t0
        if best_weights is not None:
            self.model.set_weights(best_weights)

        return self.history

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

        results = self.model.predict(X, batch_size=1000)

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
    
            loss_s = self.discriminator_loss(y_s, pred_s)
            loss_t = self.discriminator_loss(y_t, pred_t)
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
            t_loss = self.discriminator_loss(target_labels, pred_t)
    
        grads = tape.gradient(t_loss, self.target_encoder.trainable_variables)
        self.t_optimizer.apply_gradients(zip(grads, self.target_encoder.trainable_variables))
    
        return t_loss
