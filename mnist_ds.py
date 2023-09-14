from tensorflow import keras
import tensorflow as tf
import os
import numpy as np

MNIST_PATH = "./mnist.npz"


def load_mnist(path):
    if os.path.isfile(path):
        with np.load(path, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
    return keras.datasets.mnist.load_data(MNIST_PATH)


def get_half_batch_ds(batch_size):
    return get_ds(batch_size//2)


def get_ds(batch_size):
    (x, y), _ = load_mnist(MNIST_PATH)
    x = _process_x(x)
    y = tf.cast(y, tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).cache().shuffle(1024).batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_test_x():
    (_, _), (x, _) = load_mnist(MNIST_PATH)
    x = _process_x(x)
    return x


def get_test_69():
    _, (x, y) = load_mnist(MNIST_PATH)
    return _process_x(x[y == 6]), _process_x(x[y == 9])


def get_train_x():
    (x, _), _ = load_mnist(MNIST_PATH)
    x = _process_x(x)
    return x


def _process_x(x):
    return tf.expand_dims(tf.cast(x, tf.float32), axis=3) / 255. * 2 - 1


def get_69_ds():
    (x, y), _ = load_mnist(MNIST_PATH)
    x6, x9 = x[y == 6], x[y == 9]
    return _process_x(x6), _process_x(x9)


def downsampling(imgs, to_shape):
    s = to_shape[:2]
    imgs = tf.random.normal(imgs.shape, 0, 0.2) + imgs
    return tf.image.resize(imgs, size=s)