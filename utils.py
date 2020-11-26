import tensorflow as tf
import os

_b_acc = None
_c_acc = None


def set_soft_gpu(soft_gpu):
    if soft_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def binary_accuracy(label, pred):
    global _b_acc
    if _b_acc is None:
        _b_acc = tf.keras.metrics.BinaryAccuracy()
    _b_acc.reset_states()
    _b_acc.update_state(label, pred)
    return _b_acc.result()


def class_accuracy(label, pred):
    global _c_acc
    if _c_acc is None:
        _c_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    _c_acc.reset_states()
    _c_acc.update_state(label, pred)
    return _c_acc.result()


def save_weights(model):
    name = model.__class__.__name__.lower()
    os.makedirs("./models/{}".format(name), exist_ok=True)
    model.save_weights("./models/{}/model.ckpt".format(name))
