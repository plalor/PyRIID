import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Activation
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType, SpectraState, read_hdf
from riid.models.base import ModelInput, PyRIIDModel
from sklearn.metrics import accuracy_score, pairwise_distances
from time import perf_counter as time


class ADDA(PyRIIDModel):
    """Adversarial Discriminative Domain Adaptation classifier"""            
    def __init__(self, activation=None, d_optimizer=None, t_optimizer=None, warmup_optimizer = None,
                 source_model=None, discriminator_hidden_layers=None):
        """
        Args:
            activation: activation function to use for discriminator dense layer
            d_optimizer: tensorflow optimizer or optimizer name to use for training discriminator
            t_optimizer: tensorflow optimizer or optimizer name to use for training target encoder
            warmup_optimizer: tensorflow optimizer for pretraining discriminator
            source_model: pretrained source model
            discriminator_hidden_layers: size of the dense layer(s) in the discriminator
        """
        super().__init__()

        self.activation = activation
        self.d_optimizer = d_optimizer
        self.t_optimizer = t_optimizer
        self.warmup_optimizer = warmup_optimizer
        self.discriminator_hidden_layers = discriminator_hidden_layers
        self.discriminator_loss = BinaryCrossentropy()

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
            
        if self.activation is None:
            self.activation = "relu"
        if self.d_optimizer is None:
            self.d_optimizer = Adam(learning_rate=0.001)
        if self.t_optimizer is None:
            self.t_optimizer = Adam(learning_rate=0.001)
        if self.warmup_optimizer is None:
            self.warmup_optimizer = Adam(learning_rate=0.001)

        self.discriminator = None
        self.model = None

    def fit(self, source_ss: SampleSet, target_ss: SampleSet, source_val_ss: SampleSet, target_val_ss: SampleSet,
            warmup_epochs: int = 10, batch_size: int = 200, epochs: int = 20, target_level="Isotope", verbose: bool = False):
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
            warmup_epochs: number of epochs to pretrain the discriminator
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
        domain_val = tf.concat([domain_src_val, domain_tgt_val], axis=0)

        # Make datasets
        n_s = len(X_source) // batch_size
        n_t = len(X_target) // batch_size
        
        source_dataset = tf.data.Dataset.from_tensor_slices((X_source, domain_source))
        target_dataset = tf.data.Dataset.from_tensor_slices((X_target, domain_target))

        if n_s < n_t:
            source_dataset = source_dataset.repeat().shuffle(len(X_source)).batch(batch_size)
            target_dataset = target_dataset.shuffle(len(X_target)).batch(batch_size)
        elif n_t < n_s:
            source_dataset = source_dataset.shuffle(len(X_source)).batch(batch_size)
            target_dataset = target_dataset.repeat().shuffle(len(X_target)).batch(batch_size)
        else:
            source_dataset = source_dataset.shuffle(len(X_source)).batch(batch_size)
            target_dataset = target_dataset.shuffle(len(X_target)).batch(batch_size)

        ### Build discriminator
        if not self.discriminator:
            input_shape = self.source_encoder.output_shape[1]
            inputs = Input(shape=(input_shape,), name="features")
            x = inputs
            for layer, nodes in enumerate(self.discriminator_hidden_layers):
                x = Dense(nodes, activation=self.activation, name=f"dense_{layer}")(x)
            output = Dense(1, activation="sigmoid", name="discriminator")(x)
            self.discriminator = Model(inputs, output)

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

        # Warmup the discriminator
        if warmup_epochs > 0:
            if verbose:
                print("Warming up discriminator:")
            d_optimizer = self.d_optimizer # save for later
            self.d_optimizer = self.warmup_optimizer # use this optimizer for warming up
            for epoch in range(warmup_epochs):
                if verbose:
                    print(f"Epoch {epoch+1}/{warmup_epochs}")
                    t1 = time()
    
                for (x_s, y_s), (x_t, y_t) in zip(source_dataset, target_dataset):
                    d_loss = self.train_discriminator_step(x_s, x_t, y_s, y_t)
    
                f_s = self.source_encoder(X_src_val, training=False)
                f_t = self.target_encoder(X_tgt_val, training=False)
        
                pred_s = self.discriminator(f_s, training=False)
                pred_t = self.discriminator(f_t, training=False)
    
                domain_preds_s = np.round(pred_s).astype(int).flatten()
                domain_preds_t = np.round(pred_t).astype(int).flatten()
                
                domain_acc_s = accuracy_score(domain_src_val, domain_preds_s)
                domain_acc_t = accuracy_score(domain_tgt_val, domain_preds_t)
    
                if verbose:
                    print(f"Finished in {time()-t1:.0f} seconds")
                    print(f"  d_loss: {d_loss:.4f} - domain_acc_s: {domain_acc_s:.4f} - domain_acc_t: {domain_acc_t:.4f}")
                    
            self.d_optimizer = d_optimizer # restore d_optimizer for training loop

        # Training loop
        self.history = {"d_loss": [], "t_loss": [], "src_val_loss": [], "tgt_val_loss": [],
                        "d_val_loss": [], "coral": [], "entropy": []}

        best_val = np.inf
        best_weights = None
        t0 = time()
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
                t1 = time()

            for (x_s, y_s), (x_t, y_t) in zip(source_dataset, target_dataset):
                d_loss = self.train_discriminator_step(x_s, x_t, y_s, y_t)
                t_loss = self.train_target_encoder_step(x_t)

            Y_src_pred = self.model.predict(X_src_val, batch_size=batch_size)
            src_val_loss = self.model.loss(isotope_src_val, Y_src_pred).numpy()
            
            Y_tgt_pred = self.model.predict(X_tgt_val, batch_size=batch_size)
            tgt_val_loss = self.model.loss(isotope_tgt_val, Y_tgt_pred).numpy()
            
            f_s_val = self.source_encoder.predict(X_src_val, batch_size=batch_size)
            f_t_val = self.target_encoder.predict(X_tgt_val, batch_size=batch_size)
            coral = self.coral_loss(f_s_val, f_t_val)

            f_s = tf.concat([f_s_val, f_t_val], axis=0)
            pred = self.discriminator(f_s, training=False)
            d_val_loss = self.discriminator_loss(domain_val, pred)

            H = -np.sum(Y_tgt_pred * np.log(Y_tgt_pred + 1e-8), axis=1)
            entropy_tgt = H.mean()

            self.history["d_loss"].append(float(d_loss))
            self.history["t_loss"].append(float(t_loss))
            self.history["src_val_loss"].append(src_val_loss)
            self.history["tgt_val_loss"].append(tgt_val_loss)
            self.history["d_val_loss"].append(d_val_loss)
            self.history["coral"].append(coral)
            self.history["entropy"].append(entropy_tgt)
        
            # save best model weights based on validation score
            if tgt_val_loss < best_val:
                best_val = tgt_val_loss
                best_weights = self.model.get_weights()

            if verbose:
                print(f"Finished in {time()-t1:.0f} seconds")
                print(f"  d_loss: {d_loss:.4f} - t_loss: {t_loss:.4f} - src_val_loss: {src_val_loss:.4f} "
                      f"- tgt_val_loss: {tgt_val_loss:.4f}")

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

    @staticmethod
    def coral_loss(source_features, target_features):
        """
        Computes CORAL loss between source and target features.
        Both are [batch_size, feature_dim] Tensors.
        """
        s_mean = tf.reduce_mean(source_features, axis=0, keepdims=True)
        t_mean = tf.reduce_mean(target_features, axis=0, keepdims=True)
        s_centered = source_features - s_mean
        t_centered = target_features - t_mean

        cov_source = tf.matmul(tf.transpose(s_centered), s_centered)
        cov_target = tf.matmul(tf.transpose(t_centered), t_centered)

        return tf.reduce_mean(tf.square(cov_source - cov_target))

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
