# [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)

import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, save_weights
from mnist_ds import get_half_batch_ds
from gan_cnn import mnist_uni_disc_cnn
import time
import numpy as np


class AdaNorm(keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def call(self, x, training=None):
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) * (tf.math.rsqrt(ins_sigma + self.epsilon))
        return x_ins


class AdaMod(keras.layers.Layer):
    def __init__(self, trainable=None):
        super().__init__()
        self.trainable = trainable
        self.ys, self.yb = None, None

    def build(self, input_shape):
        x_input_shape, w_input_shape = input_shape
        self.ys = keras.Sequential([
            keras.layers.Dense(x_input_shape[-1], input_shape=w_input_shape[1:], trainable=self.trainable),
            keras.layers.Reshape([1, 1, -1])
        ])
        self.yb = keras.Sequential([
            keras.layers.Dense(x_input_shape[-1], input_shape=w_input_shape[1:], trainable=self.trainable),
            keras.layers.Reshape([1, 1, -1])
        ])  # [1, 1, c] per feature map

    def call(self, inputs, training=None):
        x, w = inputs
        o = self.ys(w, training=training) * x + self.yb(w, training=training)
        return o


class StyleGAN(keras.Model):
    """
    重新定义generator,生成图片
    """
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.b_scale_count = 0

        self.const = self.add_weight("const", [7, 7, 128], initializer=keras.initializers.RandomNormal(0, 0.05))
        self.f = self._get_f()
        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs[0], np.ndarray):
            inputs = (tf.convert_to_tensor(i) for i in inputs)
        return self.g.call(inputs, training=training)

    @staticmethod
    def _get_f():
        f = keras.Sequential([
            keras.layers.Dense(128),
            keras.layers.Dense(128),
        ])
        return f

    def _get_generator(self):
        z1 = keras.Input((self.latent_dim,))
        z2 = keras.Input((self.latent_dim,))
        noise_ = keras.Input((self.img_shape[0], self.img_shape[1]))
        noise = tf.expand_dims(noise_, axis=-1)

        w1 = self.f(z1)
        w2 = self.f(z2)

        x = self.add_noise(self.const, noise)
        x = AdaNorm()(x)
        x = self.style_block(64, x, w1, noise, upsampling=False)  # [7, 7]
        x = self.style_block(32, x, w1, noise)    # [14, 14]
        x = self.style_block(32, x, w2, noise)    # [28, 28]
        o = keras.layers.Conv2D(1, 5, 1, "same", activation=keras.activations.tanh)(x)
        g = keras.Model([z1, z2, noise_], o, name="generator")
        g.summary()
        return g

    def style_block(self, filters, x, w, b_noise, upsampling=True):
        x = AdaMod()((x, w))
        if upsampling:
            x = keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)

        x = keras.layers.Conv2D(filters, 3, 1, "same")(x)
        x = self.add_noise(x, b_noise)
        x = keras.layers.ReLU()(x)
        x = AdaNorm()(x)
        return x

    def add_noise(self, x, b_noise):
        x_shape = x.shape[1:] if x.shape[0] is None else x.shape
        b_noise_ = b_noise[:, :x_shape[0], :x_shape[1], :]
        scale = self.add_weight(name="b_scale{}".format(self.b_scale_count), shape=[1, 1, x.shape[-1]])
        self.b_scale_count += 1
        return scale * b_noise_ + x

    def _get_discriminator(self):
        model = keras.Sequential([
            mnist_uni_disc_cnn(self.img_shape),
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
        z1 = tf.random.normal((n, self.latent_dim))
        z2 = tf.random.normal((n, self.latent_dim)) if np.random.random() < 0.5 else z1
        noise = tf.random.normal((n, self.img_shape[0], self.img_shape[1]))
        inputs = (z1, z2, noise)
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
