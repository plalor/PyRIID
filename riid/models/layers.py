# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains custom Keras layers."""
import tensorflow as tf
from keras.api.layers import Layer
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="ReduceSumLayer")
class ReduceSumLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, axis):
        return tf.reduce_sum(x, axis=axis)


@register_keras_serializable(package="Custom", name="ReduceMaxLayer")
class ReduceMaxLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.reduce_max(x)


@register_keras_serializable(package="Custom", name="DivideLayer")
class DivideLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.divide(x[0], x[1])


@register_keras_serializable(package="Custom", name="ExpandDimsLayer")
class ExpandDimsLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, axis):
        return tf.expand_dims(x, axis=axis)


@register_keras_serializable(package="Custom", name="ClipByValueLayer")
class ClipByValueLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, clip_value_min, clip_value_max):
        return tf.clip_by_value(x, clip_value_min=clip_value_min, clip_value_max=clip_value_max)


@register_keras_serializable(package="Custom", name="PoissonLogProbabilityLayer")
class PoissonLogProbabilityLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        exp, value = x
        log_probas = tf.math.xlogy(value, exp) - exp - tf.math.lgamma(value + 1)
        return log_probas


@register_keras_serializable(package="Custom", name="SeedLayer")
class SeedLayer(Layer):
    def __init__(self, seeds, **kwargs):
        super(SeedLayer, self).__init__(**kwargs)
        self.seeds = tf.convert_to_tensor(seeds)

    def get_config(self):
        config = super().get_config()
        config.update({
            "seeds": self.seeds.numpy().tolist(),
        })
        return config

    def call(self, inputs):
        return self.seeds


@register_keras_serializable(package="Custom", name="L1NormLayer")
class L1NormLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        sums = tf.reduce_sum(inputs, axis=-1)
        l1_norm = inputs / tf.reshape(sums, (-1, 1))
        return l1_norm


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


@register_keras_serializable(package="Custom", name="GradientReversalLayer")
class GradientReversalLayer(Layer):
    def __init__(self, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lmbda = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        @tf.custom_gradient
        def _reverse_gradient(x):
            def grad(dy):
                return -self.lmbda * dy
            return x, grad
        return _reverse_gradient(inputs)
