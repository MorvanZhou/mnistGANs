from tensorflow import keras
import tensorflow as tf


def get_ds(batch_size):
    (x, y), _ = keras.datasets.mnist.load_data()
    x = _process_x(x)
    y = tf.cast(y, tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).cache().shuffle(1024).batch(batch_size//2)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_train_x():
    (x, _), _ = keras.datasets.mnist.load_data()
    x = _process_x(x)
    return x


def _process_x(x):
    return tf.expand_dims(tf.cast(x, tf.float32), axis=3) / 255. * 2 - 1