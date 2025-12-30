import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="zscore")
def zscore(x, eps=1e-12):
    m = tf.reduce_mean(x, axis=-1, keepdims=True)
    s = tf.math.reduce_std(x, axis=-1, keepdims=True)
    return (x - m) / (s + eps)


@register_keras_serializable(package="Custom", name="sqrt_zscore")
def sqrt_zscore(x, eps=1e-12):
    y = tf.sqrt(tf.maximum(x, 0.0))
    m = tf.reduce_mean(y, axis=-1, keepdims=True)
    s = tf.math.reduce_std(y, axis=-1, keepdims=True)
    z = (y - m) / (s + eps)
    return z


@register_keras_serializable(package="Custom", name="add_channel")
def add_channel(inputs):
    return tf.expand_dims(inputs, -1)


@register_keras_serializable(package="Custom", name="extract_patches")
def extract_patches(x, patch_size, stride):
    return tf.signal.frame(
        x,
        frame_length=patch_size,
        frame_step=stride,
        axis=1
    )

@register_keras_serializable(package="Custom", name="poisson_resample")
def poisson_resample(x, effective_counts):
    lam = effective_counts * x
    c = tf.random.poisson(shape=[], lam=lam)
    return tf.cast(c, tf.float32)  


@register_keras_serializable(package="Custom", name="add_sinusoidal_pos")
def add_sinusoidal_pos(x, num_patches, embed_dim):
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
    batch = tf.shape(x)[0]
    idx = tf.range(num_patches, dtype=tf.int32)[tf.newaxis, :]
    return tf.broadcast_to(idx, [batch, num_patches])


@register_keras_serializable(package="Custom", name="take_cls_token_fn")
def take_cls_token_fn(x):
    return x[:, 0, :]


def modify_dropout_rate(layer, dropout_rate):
    """Modify dropout rate for specific layer types that have dropout parameters."""
    from tensorflow.keras.layers import Dropout, SpatialDropout1D, MultiHeadAttention, LSTM
    
    if isinstance(layer, (Dropout, SpatialDropout1D)):
        config = layer.get_config()
        config['rate'] = dropout_rate
    
    elif isinstance(layer, MultiHeadAttention):
        config = layer.get_config()
        config['dropout'] = dropout_rate
    
    elif isinstance(layer, LSTM):
        config = layer.get_config()
        config['dropout'] = dropout_rate
    
    else:
        config = layer.get_config()

    return layer.__class__.from_config(config)


def clone_optimizer(opt):
    return tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(opt))
