# [Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318.pdf)

import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu, save_weights
from visual import save_gan, cvt_gif
from mnist_ds import get_half_batch_ds
import time


class Attention(keras.layers.Layer):
    def __init__(self, gamma=0.01, trainable=None):
        super().__init__(trainable=trainable)
        self._gamma = gamma
        self.gamma = None
        self.f = None
        self.g = None
        self.h = None
        self.attention = None

    def build(self, input_shape):
        self.f = self.block(input_shape[-1]//8)     # reduce channel size, reduce computation
        self.g = self.block(input_shape[-1]//8)     # reduce channel size, reduce computation
        self.h = self.block(input_shape[-1])        # scale back to original channel size
        self.gamma = tf.Variable(self._gamma)

    @staticmethod
    def block(c):
        return keras.Sequential([
            keras.layers.Conv2D(c, 1, strides=1),   # [n, w, h, c]
            keras.layers.Reshape((-1, c)),          # [n, w*h, c]
        ])

    def call(self, inputs, **kwargs):
        f = self.f(inputs)    # [n, w, h, c] -> [n, w*h, c]
        g = self.g(inputs)    # [n, w, h, c] -> [n, w*h, c]
        h = self.h(inputs)    # [n, w, h, c] -> [n, w*h, c]
        s = tf.matmul(f, g, transpose_b=True)   # [n, w*h, c] @ [n, c, w*h] = [n, w*h, w*h]
        self.attention = tf.nn.softmax(s, axis=-1)
        context_wh = tf.matmul(self.attention, h)  # [n, w*h, w*h] @ [n, w*h, c] = [n, w*h, c]
        context = tf.reshape(context_wh, tf.shape(inputs))    # [n, w, h, c]
        o = self.gamma * context + inputs
        return o


class SAGAN(keras.Model):
    """
    自注意力加强生成器能力,使用常用在SVM中的 hinge loss, 连续性loss.

    因为注意力的矩阵很大(w*h @ w*h), 所以训练起来比较慢, 意味着留有改动空间.
    里面的稳定W gradient的Spectral normalization（SN）写起来有点麻烦,
    我有空再考虑把这个 SN regularizer 写进来.
    """
    def __init__(self, latent_dim, img_shape, gamma):
        super().__init__()
        self.gamma = gamma
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.g = self._get_generator()
        self.d = self._get_discriminator()
        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_func = keras.losses.Hinge()

    def call(self, n, training=None, mask=None):
        return self.g.call(tf.random.normal((n, self.latent_dim)), training=training)

    def _get_discriminator(self):
        model = keras.Sequential([
            keras.layers.GaussianNoise(0.01, input_shape=self.img_shape),
            keras.layers.Conv2D(16, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            Attention(self.gamma),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(32, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.3),

            keras.layers.Flatten(),
            keras.layers.Dense(1),
        ], name="discriminator")
        model.summary()
        return model

    def _get_generator(self):
        model = keras.Sequential([
            # [n, latent] -> [n, 7 * 7 * 128] -> [n, 7, 7, 128]
            keras.layers.Dense(7 * 7 * 128, input_shape=(self.latent_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Reshape((7, 7, 128)),

            # -> [n, 14, 14, 64]
            keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            Attention(self.gamma),

            # -> [n, 28, 28, 32]
            keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            Attention(self.gamma),
            # -> [n, 28, 28, 1]
            keras.layers.Conv2D(1, (4, 4), padding='same', activation=keras.activations.tanh)
        ], name="generator")
        model.summary()
        return model

    def train_d(self, img, d_label):
        with tf.GradientTape() as tape:
            pred = self.d.call(img, training=True)
            loss = self.loss_func(d_label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss

    def train_g(self, d_label):
        with tf.GradientTape() as tape:
            g_img = self.call(len(d_label), training=True)
            pred = self.d.call(g_img, training=False)
            loss = self.loss_func(d_label, pred)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img

    def step(self, img):
        d_label = 2*tf.ones((len(img) * 2, 1), tf.float32)  # a stronger positive label?
        g_loss, g_img = self.train_g(d_label)

        d_label = tf.concat((tf.ones((len(img), 1), tf.float32), -tf.ones((len(g_img)//2, 1), tf.float32)), axis=0)
        img = tf.concat((img, g_img[:len(g_img)//2]), axis=0)
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
    GAMMA = 0.5
    EPOCH = 20

    set_soft_gpu(True)
    d = get_half_batch_ds(BATCH_SIZE)
    m = SAGAN(LATENT_DIM, IMG_SHAPE, GAMMA)
    train(m, d, EPOCH)
