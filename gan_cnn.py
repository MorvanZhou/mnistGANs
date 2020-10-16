from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Reshape, Conv2DTranspose, ReLU, BatchNormalization, LeakyReLU
from tensorflow import keras


def mnist_uni_gen_cnn(input_shape):
    return keras.Sequential([
        # [n, latent] -> [n, 7 * 7 * 128] -> [n, 7, 7, 128]
        Dense(7 * 7 * 128, input_shape=input_shape),
        BatchNormalization(),
        ReLU(),
        Reshape((7, 7, 128)),
        # -> [n, 14, 14, 64]
        Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        ReLU(),
        # -> [n, 28, 28, 32]
        Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        ReLU(),
        # -> [n, 28, 28, 1]
        Conv2D(1, (4, 4), padding='same', activation=keras.activations.tanh)
    ])


def mnist_uni_disc_cnn(input_shape=(28, 28, 1), use_bn=True):
    model = keras.Sequential()
    # [n, 28, 28, n] -> [n, 14, 14, 64]
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU())
    if use_bn:
        model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # -> [n, 7, 7, 128]
    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    if use_bn:
        model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Flatten())
    return model


