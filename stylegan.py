# [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)

import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, save_weights
from mnist_ds import get_half_batch_ds
from gan_cnn import mnist_uni_disc_cnn
import time
import numpy as np
import tensorflow.keras.initializers as initer


class AdaNorm(keras.layers.Layer):
    def __init__(self, axis=(1, 2), epsilon=1e-5):
        super().__init__()
        # NHWC
        self.axis = axis
        self.epsilon = epsilon

    def call(self, x, **kwargs):
        mean = tf.math.reduce_mean(x, axis=self.axis, keepdims=True)
        diff = x - mean
        variance = tf.reduce_mean(tf.math.square(diff), axis=self.axis, keepdims=True)
        x_norm = diff * tf.math.rsqrt(variance + self.epsilon)
        return x_norm


class AdaMod(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.y = None

    def call(self, inputs, **kwargs):
        x, w = inputs
        y = self.y(w)
        o = (y[:, 0] + 1) * x + y[:, 1]
        return o

    def build(self, input_shape):
        x_shape, w_shape = input_shape
        self.y = keras.Sequential([
            keras.layers.Dense(x_shape[-1]*2, input_shape=w_shape[1:], name="y",
                               kernel_initializer=initer.RandomNormal(0, 1)),   # this kernel is important
            keras.layers.Reshape([2, 1, 1, -1])])  # [2, h, w, c]


class AddNoise(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.s = None
        self.x_shape = None

    def call(self, inputs, **kwargs):
        x, noise = inputs
        noise_ = noise[:, :self.x_shape[1], :self.x_shape[2], :]
        return self.s * noise_ + x

    def build(self, input_shape):
        self.x_shape, _ = input_shape
        self.s = self.add_weight(name="noise_scale", shape=[1, 1, self.x_shape[-1]],     # [h, w, c]
                                 initializer=initer.random_normal(0., 1.))   # large initial noise


class Map(keras.layers.Layer):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.f = None

    def call(self, inputs, **kwargs):
        w = self.f(inputs)
        return w

    def build(self, input_shape):
        self.f = keras.Sequential([
            keras.layers.Dense(self.size, input_shape=input_shape[1:]),
            # keras.layers.LeakyReLU(0.2),  # worse performance when using non-linearity in mapping
            keras.layers.Dense(self.size),
        ])


class Style(keras.layers.Layer):
    def __init__(self, filters, upsampling=True):
        super().__init__()
        self.filters = filters
        self.upsampling = upsampling
        self.ada_mod, self.ada_norm, self.add_noise, self.up, self.conv = None, None, None, None, None

    def call(self, inputs, **kwargs):
        x, w, noise = inputs
        x = self.ada_mod((x, w))
        if self.up is not None:
            x = self.up(x)
        x = self.conv(x)
        x = self.ada_norm(x)
        x = keras.layers.LeakyReLU()(x)
        x = self.add_noise((x, noise))
        return x

    def build(self, input_shape):
        self.ada_mod = AdaMod()
        self.ada_norm = AdaNorm()
        if self.upsampling:
            self.up = keras.layers.UpSampling2D((2, 2), interpolation="bilinear")
        self.add_noise = AddNoise()
        self.conv = keras.layers.Conv2D(self.filters, 3, 1, "same")


class StyleGAN(keras.Model):
    """
    重新定义generator,生成图片
    """
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.n_style = 3

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.001, beta_1=0.)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs[0], np.ndarray):
            inputs = (tf.convert_to_tensor(i) for i in inputs)
        inputs = [tf.ones((len(inputs[0]), 1)), *inputs]
        return self.g.call(inputs, training=training)

    def _get_generator(self):
        z = keras.Input((self.n_style, self.latent_dim,), name="z")
        noise_ = keras.Input((self.img_shape[0], self.img_shape[1]), name="noise")
        ones = keras.Input((1,), name="ones")

        const = keras.Sequential([
            keras.layers.Dense(7*7*128, use_bias=False, name="const"),
            keras.layers.Reshape((7, 7, 128)),
        ], name="const")(ones)

        w = Map(size=128)(z)
        noise = tf.expand_dims(noise_, axis=-1)
        x = AddNoise()((const, noise))
        x = AdaNorm()(x)
        x = Style(64, upsampling=False)((x, w[:, 0], noise))    # 7^2
        x = Style(64)((x, w[:, 1], noise))      # 14^2
        x = Style(64)((x, w[:, 2], noise))  # 28^2
        o = keras.layers.Conv2D(self.img_shape[-1], 5, 1, "same", activation=keras.activations.tanh)(x)

        g = keras.Model([ones, z, noise_], o, name="generator")
        g.summary()
        return g

    def _get_discriminator(self):
        model = keras.Sequential([
            mnist_uni_disc_cnn(self.img_shape, use_bn=True),
            keras.layers.Dense(1)
        ], name="discriminator")
        model.summary()
        return model

    def train_d(self, img, label):
        with tf.GradientTape() as tape:
            pred = self.d.call(img, training=True)
            loss = self.loss_bool(label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss

    def train_g(self, n):
        available_z = [tf.random.normal((n, 1, self.latent_dim)) for _ in range(2)]
        z = tf.concat([available_z[np.random.randint(0, len(available_z))] for _ in range(self.n_style)], axis=1)

        noise = tf.random.normal((n, self.img_shape[0], self.img_shape[1]))
        inputs = (z, noise)
        with tf.GradientTape() as tape:
            g_img = self.call(inputs, training=True)
            pred = self.d.call(g_img, training=False)
            loss = self.loss_bool(tf.ones_like(pred), pred)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img

    def step(self, img):
        g_loss, g_img = self.train_g(len(img) * 2)
        d_label = tf.concat((tf.ones((len(img), 1), tf.float32), tf.zeros((len(g_img) // 2, 1), tf.float32)), axis=0)
        img = tf.concat((img, g_img[:len(g_img) // 2]), axis=0)
        d_loss = self.train_d(img, d_label)
        return d_loss, g_loss


def train(gan, ds, epoch):
    t0 = time.time()
    for ep in range(epoch):
        for t, (img, _) in enumerate(ds):
            d_loss, g_loss = gan.step(img)
            if t % 400 == 0:
                t1 = time.time()
                print(
                    "ep={} | time={:.1f} | t={} | d_loss={:.2f} | g_loss={:.2f}".format(
                        ep, t1 - t0, t, d_loss.numpy(), g_loss.numpy(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    LATENT_DIM = 100
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 64
    EPOCH = 20

    set_soft_gpu(True)
    d = get_half_batch_ds(BATCH_SIZE)
    m = StyleGAN(LATENT_DIM, IMG_SHAPE)
    train(m, d, EPOCH)
