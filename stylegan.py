# [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)

import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.layers import InstanceNormalization
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, binary_accuracy, save_weights
from mnist_ds import get_half_batch_ds
from gan_cnn import mnist_uni_disc_cnn
import time


class StyleGAN(keras.Model):
    """
    生成图片中被遮挡的部分
    """
    def __init__(self, latent_dim, noise_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.img_shape = img_shape

        self.const = self.add_weight("const", [7, 7, 64], initializer=keras.initializers.RandomNormal(0, 0.05))
        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, n, training=None, mask=None):
        return self.g.call(tf.random.normal((n, self.latent_dim)), training=training)

    def f(self):
        f = keras.Sequential([
            keras.layers.Dense(32),
            keras.layers.Dense(32),
        ], name="f")
        return f

    def _get_generator(self):
        latent = keras.Input((self.latent_dim,))
        w = self.f()(latent)
        gb = self.g_block(32, self.const, w, upsampling=False)  # [7, 7]
        gb = self.g_block(32, gb, w)    # [14, 14]
        gb = self.g_block(32, gb, w)    # [28, 28]
        o = keras.layers.Conv2D(1, 4, 1, "same", activation=keras.activations.tanh)(gb)
        g = keras.Model(latent, o, name="generator")
        g.summary()
        return g

    def g_block(self, filters, x, w, upsampling=True):
        if upsampling:
            x = keras.layers.UpSampling2D((2, 2))(x)
        x = self.add_b_noise(x)
        x = self.adaIN(w, x)
        x = keras.layers.Conv2D(filters, 3, 1, "same")(x)
        x = self.add_b_noise(x)
        x = self.adaIN(w, x)
        return x

    def add_b_noise(self, x):
        noise = tf.random.normal(x.shape[1:] if x.shape[0] is None else x.shape, 0, 0.02)  # [w, h, c]
        s = self.add_weight(name=None, shape=[1, 1, x.shape[-1]])   # [1, 1, c]
        return s * noise + x

    def adaIN(self, w, x):
        ys, yb = self.affine_transformation(w, x)
        return ys * InstanceNormalization()(x) + yb

    @staticmethod
    def affine_transformation(w, x):
        y_dim = int(x.shape[-1]*2)
        y = tf.reshape(keras.layers.Dense(y_dim)(w), [-1, 1, 1, y_dim])     # [n, 1, 1, 2c] per feature map
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
        with tf.GradientTape() as tape:
            g_img = self.call(len(d_label), training=True)
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
    NOISE_DIM = 32
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 64
    EPOCH = 20

    set_soft_gpu(True)
    d = get_half_batch_ds(BATCH_SIZE)
    m = StyleGAN(LATENT_DIM, NOISE_DIM, IMG_SHAPE)
    train(m, d, EPOCH)
