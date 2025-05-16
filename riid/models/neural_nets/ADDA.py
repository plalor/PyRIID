import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Activation
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel
from sklearn.metrics import accuracy_score, pairwise_distances
from time import perf_counter as time


class ADDA(PyRIIDModel):
    """Adversarial Discriminative Domain Adaptation classifier"""            
    def __init__(self, activation=None, d_optimizer=None, t_optimizer=None,
                 source_model=None, discriminator_hidden_layers=None):
        """
        Args:
            activation: activation function to use for discriminator dense layer
            d_optimizer: tensorflow optimizer or optimizer name to use for training discriminator
            t_optimizer: tensorflow optimizer or optimizer name to use for training target encoder
            source_model: pretrained source model
            discriminator_hidden_layers: size of the dense layer(s) in the discriminator
        """
        super().__init__()

        self.activation = activation or "relu"
        self.d_optimizer = d_optimizer or Adam(learning_rate=0.001)
        self.t_optimizer = t_optimizer or Adam(learning_rate=0.001)
        self.discriminator_hidden_layers = discriminator_hidden_layers
        self.discriminator_loss = BinaryCrossentropy()
        #self.discriminator_loss = BinaryCrossentropy(label_smoothing=0.1) # soft labels

        if source_model is not None:
            self.classification_loss = source_model.loss

            # Remove dropout layers for stability
            def strip_dropout(layer):
                if isinstance(layer, Dropout):
                    return Activation('linear', name=layer.name)
                return layer.__class__.from_config(layer.get_config())
            
            self.source_model = clone_model(
                source_model,
                clone_function=strip_dropout
            )
            self.source_model.build(source_model.input_shape)
            self.source_model.set_weights(source_model.get_weights())
            self.source_model.compile(
                optimizer=source_model.optimizer,
                loss=source_model.loss,
                metrics=source_model.metrics
            )

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
            print("WARNING: no pretrained source model was provided")

    def fit(self, source_ss: SampleSet, target_ss: SampleSet, source_val_ss: SampleSet, target_val_ss: SampleSet,
            batch_size: int = 200, epochs: int = 20, target_level="Isotope", verbose: bool = False):
        """Fit a model to the given `SampleSet`(s).

        Args:
            source_ss: `SampleSet` of `n` training spectra from the source domain where `n` >= 1
                and the spectra are either foreground (AKA, "net") or gross.
            target_ss: `SampleSet` of `n` training spectra from the target domain where `n` >= 1
                and the spectra are either foreground (AKA, "net") or gross.
            source_val_ss: `SampleSet` of `m` validation spectra from the source domain where 
                `m` >= 1 and the spectra are either foreground (AKA, "net") or gross.
            target_val_ss: `SampleSet` of `m` validation spectra from the target domain where 
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
        X_source = source_ss.get_samples().astype("float32")
        X_target = target_ss.get_samples().astype("float32")

        X_src_val = source_val_ss.get_samples().astype("float32")
        X_tgt_val = target_val_ss.get_samples().astype("float32")

        ### Isotopic labels
        isotope_src_val = source_val_ss.sources.T.groupby(target_level, sort=False).sum().T.values.astype("float32")
        isotope_tgt_val = target_val_ss.sources.T.groupby(target_level, sort=False).sum().T.values.astype("float32")

        # Domain labels: 0 for source, 1 for target
        domain_source = np.zeros(len(X_source)).reshape(-1, 1)
        domain_target = np.ones(len(X_target)).reshape(-1, 1)
        domain_src_val = np.zeros(len(X_src_val)).reshape(-1, 1)
        domain_tgt_val = np.ones(len(X_tgt_val)).reshape(-1, 1)

        # Make datasets
        half_batch_size = batch_size // 2
        steps_per_epoch = max(
            len(X_source) // half_batch_size,
            len(X_target) // half_batch_size
        )
        
        source_dataset = (
            tf.data.Dataset
              .from_tensor_slices((X_source, domain_source))
              .shuffle(len(X_source))
              .repeat()
              .batch(half_batch_size, drop_remainder=True)
        )
        
        target_dataset = (
            tf.data.Dataset
              .from_tensor_slices((X_target, domain_target))
              .shuffle(len(X_target))
              .repeat()
              .batch(half_batch_size, drop_remainder=True)
        )
        
        dataset = (
            tf.data.Dataset
              .zip((source_dataset, target_dataset))
              .shuffle(buffer_size=steps_per_epoch)
              .prefetch(tf.data.AUTOTUNE)
        )

        ### Build discriminator
        input_shape = self.source_encoder.output_shape[1]
        inputs = Input(shape=(input_shape,), name="features")
        x = inputs
        for layer, nodes in enumerate(self.discriminator_hidden_layers):
            x = Dense(nodes, activation=self.activation, name=f"dense_{layer}")(x)
        output = Dense(1, activation="sigmoid", name="discriminator")(x)
        self.discriminator = Model(inputs, output)

        ### This is what they do in the ADDA paper
        # from tensorflow.keras.layers import LeakyReLU, Dropout
        # init = RandomNormal(mean=0.0, stddev=0.02)
        # inputs = Input(shape=(input_shape,), name="features")
        # x = inputs
        
        # for i, nodes in enumerate(self.discriminator_hidden_layers):
        #     x = Dense(nodes, kernel_initializer=init, name=f"dense_{i}")(x)
        #     x = LeakyReLU(negative_slope=0.2, name=f"leakyrelu_{i}")(x)
        #     x = Dropout(0.3, name=f"dropout_{i}")(x)
        
        # output = Dense(1, activation="sigmoid", kernel_initializer=init, name="discriminator")(x)
        ###
        
        self.discriminator = Model(inputs, output, name="Discriminator")

        # Define ADDA model using target encoder and source classifier
        self.model = Model(
            inputs=self.target_encoder.input,
            outputs=self.source_classifier(self.target_encoder.output)
        )
        self.model.compile(loss=self.classification_loss, optimizer=self.t_optimizer)

        self._update_info(
            target_level=target_level,
            model_outputs = source_ss.sources.T.groupby(target_level, sort=False).sum().T.columns.values.tolist(),
            normalization=source_ss.spectra_state,
        )

        # Training loop
        self.history = {"d_loss": [], "t_loss": [], "src_val_loss": [], "tgt_val_loss": [], "d_val_loss": []}

        best_val_loss = np.inf
        best_weights = None
        t0 = time()

        it = iter(dataset)
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
                t1 = time()

            d_loss_avg = tf.metrics.Mean()
            t_loss_avg = tf.metrics.Mean()
            for step in range(steps_per_epoch):
                (x_s, y_s), (x_t, y_t) = next(it)
                d_loss = self.train_discriminator_step(x_s, x_t, y_s, y_t)
                t_loss = self.train_target_encoder_step(x_t)
                d_loss_avg.update_state(d_loss)
                t_loss_avg.update_state(t_loss)

            src_val_loss = self.calc_loss(source_val_ss, target_level=target_level, batch_size=batch_size)
            tgt_val_loss = self.calc_loss(target_val_ss, target_level=target_level, batch_size=batch_size)

            f_src_val = self.source_encoder.predict(X_src_val, batch_size=batch_size)
            f_tgt_val = self.target_encoder.predict(X_tgt_val, batch_size=batch_size)

            domain_val = tf.concat([domain_src_val, domain_tgt_val], axis=0)
            f_val = tf.concat([f_src_val, f_tgt_val], axis=0)
            pred_val = self.discriminator(f_val, training=False)
            
            d_val_loss = self.discriminator_loss(domain_val, pred_val).numpy()

            d_loss = float(d_loss_avg.result())
            t_loss = float(t_loss_avg.result())

            self.history["d_loss"].append(d_loss)
            self.history["t_loss"].append(t_loss)
            self.history["src_val_loss"].append(src_val_loss)
            self.history["tgt_val_loss"].append(tgt_val_loss)
            self.history["d_val_loss"].append(d_val_loss)
        
            # save best model weights based on validation score
            if tgt_val_loss < best_val_loss:
                best_val_loss = tgt_val_loss
                best_weights = self.model.get_weights()

            if verbose:
                print(f"Finished in {time()-t1:.0f} seconds")
                print("  "
                      f"d_loss: {d_loss:.3g} - "
                      f"t_loss: {t_loss:.3g} - "
                      f"src_val_loss: {src_val_loss:.3g} - "
                      f"tgt_val_loss: {tgt_val_loss:.3g} - "
                      f"d_val_loss: {d_val_loss:.3g}")

        self.history["training_time"] = time() - t0
        if best_weights is not None:
            self.model.set_weights(best_weights)

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
