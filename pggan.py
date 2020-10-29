# [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](https://arxiv.org/pdf/1710.10196.pdf)

import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, save_weights
from mnist_ds import get_half_batch_ds
import time


def fade(src_from, src_to, p):
    if src_from is None:
        return src_to
    if p >= 1:
        return src_to
    source = src_to * p + src_from * (1 - p)
    return source


class Generator(keras.Model):
    def __init__(self, latent_dim):
        super().__init__(name="generator")
        self.latent_dim = latent_dim

        self.upsample = keras.layers.UpSampling2D((2, 2))
        self.b1 = keras.Sequential([
            # [n, latent] -> [n, 7 * 7 * 128] -> [n, 7, 7, 128]
            keras.layers.Dense(7 * 7 * 128, input_shape=(latent_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Reshape((7, 7, 128)),
        ])
        self.b1_rgb = keras.layers.Conv2D(1, 1, 1, activation=keras.activations.tanh, input_shape=(7, 7, 128))
        self.p2, self.b2, self.b2_rgb = self._get_block(32, (7, 7, 128))
        self.p3, self.b3, self.b3_rgb = self._get_block(16, (14, 14, 32))

    @staticmethod
    def _get_block(filters, input_shape):
        project = keras.layers.Conv2D(filters, 1, 1, input_shape=input_shape)
        in_shape = input_shape[:2] + (filters,)
        block = keras.Sequential([
            keras.layers.Conv2DTranspose(filters, (4, 4), strides=2, padding='same', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        to_rgb = keras.layers.Conv2D(1, 1, 1, activation=keras.activations.tanh, input_shape=in_shape)
        return project, block, to_rgb

    def call(self, inputs, training=None, mask=None):
        current_layer, p, x = inputs
        x = self.b1(x)
        _x = None
        if current_layer >= 1:
            _x = self.upsample(x)
            _x = self.p2(_x)
            x = self.b2(x)  # better to learn downsampling?
        if current_layer >= 2:
            _x = self.upsample(x)
            _x = self.p3(_x)
            x = self.b3(x)
        if current_layer >= 3:
            _x = None
        _x = self.to_rgb(current_layer, _x)
        x = self.to_rgb(current_layer, x)
        x = fade(_x, x, p)
        return x

    def to_rgb(self, current_layer, x):
        if x is None:
            return x
        if current_layer == 0:
            return self.b1_rgb(x)
        elif current_layer == 1:
            return self.b2_rgb(x)
        else:
            return self.b3_rgb(x)


class Discriminator(keras.Model):
    def __init__(self):
        super().__init__(name="discriminator")
        self.b3_rgb, self.b3, self.p3 = self._get_block(32, (28, 28, 1))
        self.b2_rgb, self.b2, self.p2 = self._get_block(64, (14, 14, 32))
        self.b1_rgb = keras.layers.Conv2D(64, 1, 1, activation=keras.activations.tanh, input_shape=(7, 7, 128))
        self.b1 = keras.Sequential([
            keras.layers.Flatten(input_shape=(7, 7, 64)),
            keras.layers.Dense(1)
        ])
        self.pool = keras.layers.AvgPool2D((4, 4), strides=(2, 2), padding="SAME")

    @staticmethod
    def _get_block(filters, input_shape):
        from_rgb = keras.layers.Conv2D(
            input_shape[-1], 1, 1, activation=keras.activations.tanh, input_shape=input_shape[:2] + (1,))
        block = keras.Sequential([
            keras.layers.Conv2D(filters, 4, strides=2, padding='same', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.3),
        ])
        project = keras.layers.Conv2D(filters, 1, 1, input_shape=input_shape)
        return from_rgb, block, project

    def call(self, inputs, training=None, mask=None):
        current_layer, p, x = inputs
        x = self.from_rgb(current_layer, x)
        _x = None if current_layer >= 3 or current_layer == 0 else self.pool(x)

        if current_layer >= 2:
            x = self.b3(x)   # better to learn upsampling?
            if current_layer == 2:
                _x = self.p3(_x)
                x = fade(_x, x, p)
        if current_layer >= 1:
            x = self.b2(x)
            if current_layer == 1:
                _x = self.p2(_x)
                x = fade(_x, x, p)
        o = self.b1(x)
        return o

    def project(self, current_layer, x):
        if x is None:
            return None
        if current_layer == 0:
            return x
        elif current_layer == 1:
            return self.p2(x)
        elif current_layer == 2:
            return self.p3(x)
        return x

    def from_rgb(self, current_layer, x):
        if current_layer == 0:
            return self.b1_rgb(x)
        elif current_layer == 1:
            return self.b2_rgb(x)
        else:
            return self.b3_rgb(x)


class PGGAN(keras.Model):
    def __init__(self, latent_dim, img_shape, fade_in_step=1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.fade_in_step = fade_in_step
        self.fade_in_count = 0
        self.current_layer = 0

        self.g = Generator(latent_dim)
        self.d = Discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_func = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, n, training=None, mask=None):
        inputs = [
            self.current_layer,
            self.fade_in_count / self.fade_in_step,
            tf.random.normal((n, self.latent_dim)),
        ]
        return self.g(inputs, training=training)

    def train_d(self, img, label):
        inputs = [self.current_layer, self.fade_in_count / self.fade_in_step, img]
        with tf.GradientTape() as tape:
            pred = self.d(inputs, training=True)
            loss = self.loss_func(label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss

    def train_g(self, d_label):
        with tf.GradientTape() as tape:
            g_img = self.call(len(d_label), training=True)
            pred = self.d([self.current_layer, self.fade_in_count / self.fade_in_step, g_img], training=False)
            loss = self.loss_func(d_label, pred)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img

    def step(self, img):
        d_label = tf.ones((len(img) * 2, 1), tf.float32)  # let d think generated images are real
        g_loss, g_img = self.train_g(d_label)

        d_label = tf.concat(
            (tf.ones((len(img), 1), tf.float32), tf.zeros((len(g_img)//2, 1), tf.float32)), axis=0)
        real_fake_img = tf.concat((self.resize_img(img, g_img), g_img[:len(g_img)//2]), axis=0)
        d_loss = self.train_d(real_fake_img, d_label)

        # count
        self.fade_in_count += 1
        if self.fade_in_count >= self.fade_in_step * 1.5:
            self.fade_in_count = 0
            self.current_layer = min(self.current_layer + 1, 3)
        return d_loss, g_loss

    @staticmethod
    def resize_img(img, g_img):
        return tf.image.resize(img, tf.shape(g_img)[1:3], tf.image.ResizeMethod.NEAREST_NEIGHBOR)


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
    FADE_STEP = 2000
    EPOCH = 8

    set_soft_gpu(True)
    d = get_half_batch_ds(BATCH_SIZE)
    m = PGGAN(LATENT_DIM, IMG_SHAPE, FADE_STEP)
    train(m, d, EPOCH)



