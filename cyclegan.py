# [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593)

import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, save_weights
from mnist_ds import get_69_ds, get_test_69
from gan_cnn import mnist_uni_disc_cnn, mnist_uni_img2img
import time
import numpy as np


class CycleGAN(keras.Model):
    """
    在两种不同风格的图片中来回迁移
    """
    def __init__(self, lambda_, img_shape, use_identity=False):
        super().__init__()
        self.lambda_ = lambda_
        self.img_shape = img_shape
        self.use_identity = use_identity

        self.g12 = self._get_generator("g12")
        self.g21 = self._get_generator("g21")
        self.d12 = self._get_discriminator("d12")
        self.d21 = self._get_discriminator("d21")

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_img = keras.losses.MeanAbsoluteError()    # a better result when using mse

    def _get_generator(self, name):
        model = mnist_uni_img2img(self.img_shape, name=name, norm="instance")
        model.summary()
        return model

    def _get_discriminator(self, name):
        model = keras.Sequential([
            keras.layers.GaussianNoise(0.01, input_shape=self.img_shape),   # add some noise
            mnist_uni_disc_cnn(),
            keras.layers.Dense(1)
        ], name=name)
        model.summary()
        return model

    def train_d(self, real_fake1, real_fake2, label):
        with tf.GradientTape() as tape:
            loss = self.loss_bool(label, self.d12(real_fake2)) + self.loss_bool(label, self.d21(real_fake1))
        var = self.d12.trainable_variables + self.d21.trainable_variables
        grads = tape.gradient(loss, var)
        self.opt.apply_gradients(zip(grads, var))
        return loss

    def cycle(self, real1, real2):
        fake2, fake1 = self.g12(real1), self.g21(real2)
        loss1 = self.loss_img(real1, self.g21(fake2))
        loss2 = self.loss_img(real2, self.g12(fake1))
        return loss1 + loss2, fake2, fake1

    def identity(self, real1, real2):
        loss21 = self.loss_img(real1, self.g21(real2))
        loss12 = self.loss_img(real2, self.g12(real1))
        return loss21, loss12

    def train_g(self, real1, real2):
        with tf.GradientTape() as tape:
            cycle_loss, fake2, fake1 = self.cycle(real1, real2)
            pred12 = self.d12(fake2)
            pred21 = self.d21(fake1)
            d_loss12 = self.loss_bool(tf.ones_like(pred12), pred12)
            d_loss21 = self.loss_bool(tf.ones_like(pred21), pred21)
            loss12 = d_loss12 + self.lambda_ * cycle_loss
            loss21 = d_loss21 + self.lambda_ * cycle_loss
            if self.use_identity:
                id_loss21, id_loss12 = self.identity(real1, real2)
                loss12 += self.lambda_ * id_loss12
                loss21 += self.lambda_ * id_loss21
            loss = loss12 + loss21
        var = self.g12.trainable_variables + self.g21.trainable_variables
        grads = tape.gradient(loss, var)
        self.opt.apply_gradients(zip(grads, var))

        half = len(real1) // 2
        return d_loss12+d_loss21, cycle_loss, fake2[:half], fake1[:half]

    def step(self, img1, img2):
        g_loss, cycle_loss, half_fake2, half_fake1 = self.train_g(img1, img2)

        half = len(half_fake1)
        d_label = tf.concat((tf.ones((half, 1), tf.float32), tf.zeros((half, 1), tf.float32)), axis=0)
        real_fake1 = tf.concat((img1[:half], half_fake1), axis=0)
        real_fake2 = tf.concat((img2[:half], half_fake2), axis=0)
        d_loss = self.train_d(real_fake1, real_fake2, d_label)
        return g_loss, d_loss, cycle_loss


def train(gan, x6, x9, test6, test9, step, batch_size):
    t0 = time.time()
    for t in range(step):
        idx6 = np.random.randint(0, len(x6), batch_size)
        img6 = tf.gather(x6, idx6)
        idx9 = np.random.randint(0, len(x9), batch_size)
        img9 = tf.gather(x9, idx9)
        g_loss, d_loss, cyc_loss = gan.step(img6, img9)
        if t % 500 == 0:
            t1 = time.time()
            print(
                "t={}|time={:.1f}|g_loss={:.2f}|d_loss={:.2f}|cyc_loss={:.2f}".format(
                    t, t1 - t0, g_loss.numpy(), d_loss.numpy(), cyc_loss.numpy()))
            t0 = t1
            save_gan(gan, t, img6=test6, img9=test9)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 32
    LAMBDA = 10
    STEP = 10001

    set_soft_gpu(True)
    X6, X9 = get_69_ds()
    TEST6, TEST9 = get_test_69()
    m = CycleGAN(LAMBDA, IMG_SHAPE)
    train(m, X6, X9, TEST6, TEST9, STEP, BATCH_SIZE)



