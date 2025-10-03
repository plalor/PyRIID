import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="zscore")
def zscore(x):
    """Z-score normalization function."""
    m = tf.reduce_mean(x, axis=-1, keepdims=True)
    s = tf.math.reduce_std(x, axis=-1, keepdims=True)
    return (x - m) / s


@register_keras_serializable(package="Custom", name="add_channel")
def add_channel(inputs):
    """Add a channel dimension to inputs."""
    return tf.expand_dims(inputs, -1)


@register_keras_serializable(package="Custom", name="extract_patches")
def extract_patches(x, patch_size, stride):
    """Extract patches from input tensor."""
    return tf.signal.frame(
        x,
        frame_length=patch_size,
        frame_step=stride,
        axis=1
    )


@register_keras_serializable(package="Custom", name="add_sinusoidal_pos")
def add_sinusoidal_pos(x, num_patches, embed_dim):
    """Add sinusoidal positional encoding."""
    pos = tf.cast(tf.range(num_patches)[:, None], tf.float32)
    i = tf.cast(tf.range(embed_dim)[None, :], tf.float32)
    angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(i/2.0)) / tf.cast(embed_dim, tf.float32))
    angle_rads = pos * angle_rates
    sin = tf.sin(angle_rads)
    cos = tf.cos(angle_rads)
    even_mask = tf.cast(tf.equal(tf.math.floormod(tf.range(embed_dim), 2), 0), tf.float32)[None, :]
    pos_encoding = sin * even_mask + cos * (1.0 - even_mask)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.broadcast_to(pos_encoding, [tf.shape(x)[0], num_patches, embed_dim])


@register_keras_serializable(package="Custom", name="make_positions")
def make_positions(x, num_patches):
    """Create position indices."""
    return tf.range(num_patches, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]


@register_keras_serializable(package="Custom", name="take_cls_token_fn")
def take_cls_token_fn(x):
    """Extract the CLS token from the sequence."""
    return x[:, 0, :]
