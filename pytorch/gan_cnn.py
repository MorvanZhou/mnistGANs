import torch
import torch.nn as nn
from einops import rearrange

# class InstanceNormalization(keras.layers.Layer):
#     def __init__(self, axis=(1, 2), epsilon=1e-6):
#         super().__init__()
#         # NHWC
#         self.epsilon = epsilon
#         self.axis = axis
#         self.beta, self.gamma = None, None

#     def build(self, input_shape):
#         # NHWC
#         shape = [1, 1, 1, input_shape[-1]]
#         self.gamma = self.add_weight(
#             name='gamma',
#             shape=shape,
#             initializer='ones')

#         self.beta = self.add_weight(
#             name='beta',
#             shape=shape,
#             initializer='zeros')

#     def call(self, x, *args, **kwargs):
#         mean = tf.math.reduce_mean(x, axis=self.axis, keepdims=True)
#         diff = x - mean
#         variance = tf.reduce_mean(tf.math.square(diff), axis=self.axis, keepdims=True)
#         x_norm = diff * tf.math.rsqrt(variance + self.epsilon)
#         return x_norm * self.gamma + self.beta


class mnist_uni_gen_cnn(nn.Module):
    def __init__(self, input_shape):
        super(mnist_uni_gen_cnn, self).__init__()

        # [n, latent] -> [n, 7 * 7 * 128] -> [n, 7, 7, 128]
        upsample = []
        upsample.append(nn.Linear(input_shape, 7*7*128))
        upsample.append(nn.BatchNorm1d(7*7*128))
        upsample.append(nn.ReLU())
        self.upsample = nn.Sequential(*upsample)

        # -> [n, 64, 14, 14]
        deconv = []
        deconv.append(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1))
        deconv.append(nn.BatchNorm2d(64))
        deconv.append(nn.ReLU())

        # -> [n, 32, 28, 28]
        deconv.append(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1))
        deconv.append(nn.BatchNorm2d(32))
        deconv.append(nn.ReLU())

        # -> [n, 1, 28, 28]
        deconv.append(nn.Conv2d(32, 1, kernel_size=3, padding=1))
        deconv.append(nn.Tanh())
        self.deconv = nn.Sequential(*deconv)

    def forward(self, x):
        x = self.upsample(x)
        x = x.reshape(x.shape[0], 128,7,7)
        x = self.deconv(x)
        return x


class mnist_uni_disc_cnn(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), use_bn=True):
        super(mnist_uni_disc_cnn, self).__init__()
        layers = []

        # [n, c, 28, 28] -> [n, 64, 14, 14]
        layers.append(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))

        # -> [n, 128, 7, 7]
        layers.append(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))

        layers.append(nn.Flatten())
        self.disc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.disc(x)
        return x



# def mnist_uni_img2img(img_shape, name="generator", norm="batch"):
#     def do_norm():
#         if norm == "batch":
#             _norm = [BatchNormalization()]
#         elif norm == "instance":
#             _norm = [InstanceNormalization()]
#         else:
#             _norm = []
#         return _norm
#     model = keras.Sequential([
#         # [n, 28, 28, n] -> [n, 14, 14, 64]
#         Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=img_shape),
#         ] + do_norm() + [
#         LeakyReLU(),
#         # -> [n, 7, 7, 128]
#         Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
#         ] + do_norm() + [
#         LeakyReLU(),

#         # -> [n, 14, 14, 64]
#         Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
#         ] + do_norm() + [
#         ReLU(),
#         # -> [n, 28, 28, 32]
#         Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
#         ] + do_norm() + [
#         ReLU(),
#         # -> [n, 28, 28, 1]
#         Conv2D(img_shape[-1], (4, 4), padding='same', activation=keras.activations.tanh)
#     ], name=name)
#     return model


# def mnist_unet(img_shape):
#     i = keras.Input(shape=img_shape, dtype=tf.float32)
#     # [n, 28, 28, n] -> [n, 14, 14, 64]
#     l1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=img_shape)(i)
#     l1 = BatchNormalization()(l1)
#     l1 = LeakyReLU()(l1)
#     # -> [n, 7, 7, 128]
#     l2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(l1)
#     l2 = BatchNormalization()(l2)
#     l2 = LeakyReLU()(l2)

#     # -> [n, 14, 14, 64]
#     u1 = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(l2)
#     u1 = BatchNormalization()(u1)
#     u1 = ReLU()(u1)
#     u1 = tf.concat((u1, l1), axis=3)    # -> [n, 14, 14, 64 + 64]
#     # -> [n, 28, 28, 32]
#     u2 = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(u1)
#     u2 = BatchNormalization()(u2)
#     u2 = ReLU()(u2)
#     u2 = tf.concat((u2, i), axis=3)     # -> [n, 28, 28, 32 + n]
#     # -> [n, 28, 28, 1]
#     o = Conv2D(img_shape[-1], (4, 4), padding='same', activation=keras.activations.tanh)(u2)

#     unet = keras.Model(i, o, name="unet")
#     return unet



