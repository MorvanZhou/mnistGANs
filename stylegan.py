# [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)

import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, binary_accuracy, save_weights
from mnist_ds import get_half_batch_ds
from gan_cnn import mnist_uni_disc_cnn
import time
import numpy as np
try:
    from tensorflow_addons.layers import InstanceNormalization
except ImportError:
    from utils import InstanceNormalization


class StyleGAN(keras.Model):
    """
    重新定义generator,生成图片
    """
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.b_scale_count = 0

        self.const = self.add_weight("const", [7, 7, 64], initializer=keras.initializers.RandomNormal(0, 0.05))
        self.f = self._get_f()
        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs[0], np.ndarray):
            inputs = (tf.convert_to_tensor(i) for i in inputs)
        return self.g.call(inputs, training=training)

    def _get_f(self):
        f = keras.Sequential([
            keras.layers.Dense(32),
            keras.layers.Dense(32),
        ])
        return f

    def _get_generator(self):
        z1 = keras.Input((self.latent_dim,))
        z2 = keras.Input((self.latent_dim,))
        noise = keras.Input((self.img_shape[0], self.img_shape[1]))
        b_noise = tf.expand_dims(noise, axis=-1)
        w1 = self.f(z1)
        w2 = self.f(z2)
        gb = self.g_block(32, self.const, w1, b_noise, upsampling=False)  # [7, 7]
        gb = self.g_block(32, gb, w1, b_noise)    # [14, 14]
        gb = self.g_block(16, gb, w2, b_noise)    # [28, 28]
        o = keras.layers.Conv2D(1, 4, 1, "same", activation=keras.activations.tanh)(gb)
        g = keras.Model([z1, z2, noise], o, name="generator")
        g.summary()
        return g

    def g_block(self, filters, x, w, b_noise, upsampling=True):
        if upsampling:
            x = keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
            x = keras.layers.Conv2D(filters, 3, 1, "same")(x)
        x_shape = x.shape[1:] if x.shape[0] is None else x.shape
        b_noise_ = b_noise[:, :x_shape[0], :x_shape[1], :]
        x = self.add_noise(x, b_noise_)
        x = self.adaIN(w, x)
        x = keras.layers.Conv2D(filters, 3, 1, "same")(x)
        x = self.add_noise(x, b_noise_)
        x = self.adaIN(w, x)
        return x

    def add_noise(self, x, b_noise):
        scale = self.add_weight(name="b_scale{}".format(self.b_scale_count), shape=[1, 1, x.shape[-1]])
        self.b_scale_count += 1
        return scale * b_noise + x

    def adaIN(self, w, x):
        ys, yb = self.affine_transformation(w, x)
        return ys * InstanceNormalization()(x) + yb

    @staticmethod
    def affine_transformation(w, x):
        y_dim = int(x.shape[-1]*2)
        y = keras.layers.Dense(y_dim)(w)
        y = keras.layers.Reshape([1, 1, -1])(y)     # [n, 1, 1, 2c] per feature map
        ys, yb = tf.split(y, 2, axis=-1)        # [n, 1, 1, c]
        return ys, yb

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
        return loss, binary_accuracy(label, pred)

    def train_g(self, d_label):
        inputs = (tf.random.normal((len(d_label), self.latent_dim)),
                  tf.random.normal((len(d_label), self.latent_dim)),
                  tf.random.normal((len(d_label), self.img_shape[0], self.img_shape[1])))
        with tf.GradientTape() as tape:
            g_img = self.call(inputs, training=True)
            pred = self.d.call(g_img, training=False)
            loss = self.loss_bool(d_label, pred)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img, binary_accuracy(d_label, pred)

    def step(self, img):
        d_label = tf.ones((len(img) * 2, 1), tf.float32)  # let d think generated images are real
        g_loss, g_img, g_acc = self.train_g(d_label)

        d_label = tf.concat((tf.ones((len(img), 1), tf.float32), tf.zeros((len(g_img) // 2, 1), tf.float32)), axis=0)
        img = tf.concat((img, g_img[:len(g_img) // 2]), axis=0)
        d_loss, d_acc = self.train_d(img, d_label)
        return d_loss, d_acc, g_loss, g_acc


def train(gan, ds, epoch):
    t0 = time.time()
    for ep in range(epoch):
        for t, (img, _) in enumerate(ds):
            d_loss, d_acc, g_loss, g_acc = gan.step(img)
            if t % 400 == 0:
                t1 = time.time()
                print(
                    "ep={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".format(
                        ep, t1 - t0, t, d_acc.numpy(), g_acc.numpy(), d_loss.numpy(), g_loss.numpy(), ))
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
