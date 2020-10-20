from tensorflow import keras
import tensorflow as tf


def get_half_batch_ds(batch_size):
    return get_ds(batch_size//2)


def get_ds(batch_size):
    (x, y), _ = keras.datasets.mnist.load_data()
    x = _process_x(x)
    y = tf.cast(y, tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).cache().shuffle(1024).batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_test_x():
    (_, _), (x, _) = keras.datasets.mnist.load_data()
    x = _process_x(x)
    return x


def get_test_69():
    _, (x, y) = keras.datasets.mnist.load_data()
    return _process_x(x[y == 6]), _process_x(x[y == 9])


def get_train_x():
    (x, _), _ = keras.datasets.mnist.load_data()
    x = _process_x(x)
    return x


def _process_x(x):
    return tf.expand_dims(tf.cast(x, tf.float32), axis=3) / 255. * 2 - 1


def get_69_ds():
    (x, y), _ = keras.datasets.mnist.load_data()
    x6, x9 = x[y == 6], x[y == 9]
    return _process_x(x6), _process_x(x9)
