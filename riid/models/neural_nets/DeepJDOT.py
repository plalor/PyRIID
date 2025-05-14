import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from riid import SampleSet, SpectraType, SpectraState, read_hdf
from riid.models.base import ModelInput, PyRIIDModel
from time import perf_counter as time


class DeepJDOT(PyRIIDModel):
    """Classifier using DeepJDOT (Deep Joint Distribution Optimal Transport) domain adaptation."""
    def __init__(self, optimizer=None, source_model=None, ot_weight=1.0, sinkhorn_reg=0.1, 
                 num_sinkhorn_iters=10, jdot_alpha=1.0, jdot_beta=1.0):
        """
        Args:
            optimizer: tensorflow optimizer or optimizer name
            source_model: Pretrained source model.
            ot_weight: Weight for the OT loss term.
            sinkhorn_reg: Entropic regularization parameter for Sinkhorn iterations.
            num_sinkhorn_iters: Number of iterations in the Sinkhorn algorithm.
            jdot_alpha: Weight for the feature-distance term in the cost matrix.
            jdot_beta: Weight for the classification loss term in the cost matrix.
        """
        super().__init__()
        self.optimizer = optimizer
        self.ot_weight = ot_weight
        self.sinkhorn_reg = sinkhorn_reg
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.jdot_alpha = jdot_alpha
        self.jdot_beta = jdot_beta

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
            feature_extractor_input = self.source_model.input
            feature_extractor_output = all_layers[-2].output
            self.feature_extractor = Model(inputs=feature_extractor_input, outputs=feature_extractor_output, name="feature_extractor")

            classifier_input = Input(shape=feature_extractor_output.shape[1:], name="feature_extractor_output")
            classifier_output = all_layers[-1](classifier_input)
            self.classifier = Model(inputs=classifier_input, outputs=classifier_output, name="classifier")
        else:
            print("WARNING: no pretrained source model was provided")

        if self.optimizer is None:
            self.optimizer = Adam(learning_rate=0.001)

        self.model = None

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
            raise ValueError("Empty spectra provided!")

        if source_ss.spectra_type == SpectraType.Gross:
            self.model_inputs = (ModelInput.GrossSpectrum,)
        elif source_ss.spectra_type == SpectraType.Foreground:
            self.model_inputs = (ModelInput.ForegroundSpectrum,)
        elif source_ss.spectra_type == SpectraType.Background:
            self.model_inputs = (ModelInput.BackgroundSpectrum,)
        else:
            raise ValueError(f"{source_ss.spectra_type} is not supported in this model.")

        ### Prepare training data
        X_source = source_ss.get_samples().astype("float32")
        X_target = target_ss.get_samples().astype("float32")

        X_src_val = source_val_ss.get_samples().astype("float32")
        X_tgt_val = target_val_ss.get_samples().astype("float32")

        source_contributions_df = source_ss.sources.T.groupby(target_level, sort=False).sum().T
        model_outputs = source_contributions_df.columns.values.tolist()
        Y_source = source_contributions_df.values.astype("float32")

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

        # Define DeepJDOT model
        self.model = Model(
            inputs=self.feature_extractor.input,
            outputs=self.classifier(self.feature_extractor.output)
        )
        self.model.compile(loss=self.classification_loss)

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs=model_outputs,
            normalization=source_ss.spectra_state,
        )

        # Training loop
        self.history = {"class_loss": [], "total_loss": [], "ot_loss": [], "tgt_val_loss": [], "ot_val_loss": []}
        best_val_loss = np.inf
        best_weights = None
        t0 = time()

        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
                t1 = time()

            total_loss_avg = tf.metrics.Mean()
            class_loss_avg = tf.metrics.Mean()
            ot_loss_avg = tf.metrics.Mean()

            for (x_s, y_s), x_t in zip(source_dataset, target_dataset):
                total_loss, class_loss, ot_loss = self.train_step(x_s, y_s, x_t)
                total_loss_avg.update_state(total_loss)
                class_loss_avg.update_state(class_loss)
                ot_loss_avg.update_state(ot_loss)

            tgt_val_loss = self.calc_loss(target_val_ss, target_level=target_level, batch_size=batch_size)

            ot_val_loss = xxx
            
            self.history["class_loss"].append(float(class_loss_avg.result()))
            self.history["total_loss"].append(float(total_loss_avg.result()))
            self.history["ot_loss"].append(float(ot_loss_avg.result()))
            self.history["tgt_val_loss"].append(tgt_val_loss)
            self.history["ot_val_loss"].append(ot_val_loss)

            # save best model weights based on validation score
            if tgt_val_loss < best_val_loss:
                best_val_loss = tgt_val_loss
                best_weights = self.model.get_weights()

            if verbose:
                print(f"Finished in {time()-t1:.0f} seconds")
                print("  "
                      f"total_loss: {total_loss_avg.result():.4f} - "
                      f"class_loss: {class_loss_avg.result():.4f} - "
                      f"ot_loss: {ot_loss_avg.result():.4f} - "
                      f"tgt_val_loss: {tgt_val_loss:.4f} - "
                      f"ot_val_loss: {ot_val_loss:.4f}")

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
    def pairwise_squared_distance(a, b):
        """
        Compute pairwise squared Euclidean distances between rows of tensors a and b.
        
        Args:
            a: Tensor of shape (n, d).
            b: Tensor of shape (m, d).
            
        Returns:
            Tensor of shape (n, m) containing squared distances.
        """
        a_norm = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)  # shape (n, 1)
        b_norm = tf.reduce_sum(tf.square(b), axis=1, keepdims=True)  # shape (m, 1)
        dist = a_norm + tf.transpose(b_norm) - 2 * tf.matmul(a, b, transpose_b=True)
        return tf.maximum(dist, 0.0)

    @staticmethod
    def pairwise_classification_loss(y_true, y_pred):
        """
        Compute pairwise classification loss (categorical crossentropy) between y_true and y_pred.
        y_true: Tensor of shape (n, num_classes)
        y_pred: Tensor of shape (m, num_classes)
        Returns: Tensor of shape (n, m)
        """
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        # Expand dims to compute loss for every source-target pair.
        y_true_exp = tf.expand_dims(y_true, axis=1)  # (n, 1, num_classes)
        y_pred_exp = tf.expand_dims(y_pred, axis=0)  # (1, m, num_classes)
        # Compute crossentropy: -sum(y_true * log(y_pred)) for each pair.
        ce = -tf.reduce_sum(y_true_exp * tf.math.log(y_pred_exp), axis=-1)  # shape (n, m)
        return ce

    @staticmethod
    def sinkhorn(a, b, M, reg, num_iters):
        """
        Compute the Sinkhorn algorithm for optimal transport.
        a: Tensor of shape (n,), source distribution (sums to 1)
        b: Tensor of shape (m,), target distribution (sums to 1)
        M: Cost matrix of shape (n, m)
        reg: Regularization parameter (epsilon)
        num_iters: Number of Sinkhorn iterations
        Returns: Optimal transport plan gamma of shape (n, m)
        """
        K = tf.exp(-M / reg)  # (n, m)
        u = tf.ones_like(a)
        v = tf.ones_like(b)

        for _ in range(num_iters):
            u = a / (tf.linalg.matvec(K, v) + 1e-8)
            v = b / (tf.linalg.matvec(tf.transpose(K), u) + 1e-8)

        gamma = tf.expand_dims(u, 1) * K * tf.expand_dims(v, 0)
        return gamma

    @tf.function
    def train_step(self, x_s, y_s, x_t):
        """
        Perform one training step:
          - Compute source features and predictions.
          - Compute target features and predictions.
          - Build a cost matrix that combines feature distances and a crossentropy term.
          - Compute the OT plan via Sinkhorn iterations.
          - Compute the OT loss and add it to the source classification loss.
        """
        with tf.GradientTape() as tape:
            # Forward pass on source
            f_s = self.feature_extractor(x_s, training=True)
            p_s = self.classifier(f_s, training=True)
            source_class_loss = self.classification_loss(y_s, p_s)

            # Forward pass on target
            f_t = self.feature_extractor(x_t, training=True)
            p_t = self.classifier(f_t, training=True)

            # Cost matrix: feature distance + classification loss term
            feat_distance = self.pairwise_squared_distance(f_s, f_t)
            cls_loss_matrix = self.pairwise_classification_loss(y_s, p_t)
            cost_matrix = self.jdot_alpha * feat_distance + self.jdot_beta * cls_loss_matrix

            # Define uniform marginals over the source and target batches
            n = tf.cast(tf.shape(x_s)[0], tf.float32)
            m = tf.cast(tf.shape(x_t)[0], tf.float32)
            a = tf.fill([tf.shape(x_s)[0]], 1.0 / n)
            b = tf.fill([tf.shape(x_t)[0]], 1.0 / m)

            # Compute the OT coupling using Sinkhorn iterations
            gamma = self.sinkhorn(a, b, cost_matrix, self.sinkhorn_reg, self.num_sinkhorn_iters)

            # OT loss is the inner product between the coupling and cost matrix
            ot_loss = tf.reduce_sum(gamma * cost_matrix)

            # Total loss: source classification loss + weighted OT loss
            total_loss = source_class_loss + self.ot_weight * ot_loss

        grads = tape.gradient(total_loss, self.source_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.source_model.trainable_variables))
        return total_loss, source_class_loss, ot_loss
        