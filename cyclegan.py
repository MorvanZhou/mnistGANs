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
    def __init__(self, lambda_, img_shape):
        super().__init__()
        self.lambda_ = lambda_
        self.img_shape = img_shape

        self.g = self._get_generator("g")
        self.f = self._get_generator("f")
        self.dg = self._get_discriminator("dg")
        self.df = self._get_discriminator("df")

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_img = keras.losses.MeanAbsoluteError()    # a better result when using mse

    def _get_generator(self, name):
        model = mnist_uni_img2img(self.img_shape, name=name)
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

    def train_d(self, fimg, gimg, label):
        with tf.GradientTape() as tape:
            loss = (self.loss_bool(label, self.dg(gimg)) + self.loss_bool(label, self.df(fimg)))/2
        vars = self.df.trainable_variables + self.dg.trainable_variables
        grads = tape.gradient(loss, vars)
        self.opt.apply_gradients(zip(grads, vars))
        return loss

    def cycle(self, img1, img2):
        gimg1, fimg2 = self.g(img1), self.f(img2)
        loss1 = self.loss_img(img1, self.f(gimg1))
        loss2 = self.loss_img(img2, self.g(fimg2))
        loss = self.lambda_ * (loss1 + loss2) / 2
        return loss, gimg1, fimg2

    def identity(self, img1, img2):
        loss1 = self.loss_img(img2, self.g(img2))
        loss2 = self.loss_img(img1, self.f(img1))
        return self.lambda_ * (loss1 + loss2) / 2

    def train_g(self, img1, img2):
        d_label = tf.ones((len(img1), 1), tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            cycle_loss, gimg1, fimg2 = self.cycle(img1, img2)
            identity_loss = self.identity(img1, img2)
            loss_g = self.loss_bool(d_label, self.dg(gimg1)) + cycle_loss + identity_loss
            loss_f = self.loss_bool(d_label, self.df(fimg2)) + cycle_loss + identity_loss
        grads_g = tape.gradient(loss_g, self.g.trainable_variables)
        grads_f = tape.gradient(loss_f, self.f.trainable_variables)
        self.opt.apply_gradients(zip(grads_g, self.g.trainable_variables))
        self.opt.apply_gradients(zip(grads_f, self.f.trainable_variables))
        del tape

        half = len(img1) // 2
        return (loss_g+loss_f)/2, gimg1[:half], fimg2[:half]

    def step(self, img1, img2):
        g_loss, gimg1, fimg2 = self.train_g(img1, img2)

        half = len(fimg2)
        d_label = tf.concat((tf.ones((half, 1), tf.float32), tf.zeros((half, 1), tf.float32)), axis=0)
        real_fake_fimg = tf.concat((img1[:half], fimg2), axis=0)
        real_fake_gimg = tf.concat((img2[:half], gimg1), axis=0)
        d_loss = self.train_d(real_fake_fimg, real_fake_gimg, d_label)
        return g_loss, d_loss


def train(gan, x6, x9, test6, test9, step, batch_size):
    t0 = time.time()
    for t in range(step):
        idx6 = np.random.randint(0, len(x6), batch_size)
        img6 = tf.gather(x6, idx6)
        idx9 = np.random.randint(0, len(x9), batch_size)
        img9 = tf.gather(x9, idx9)
        g_loss, d_loss = gan.step(img6, img9)
        if t % 500 == 0:
            t1 = time.time()
            print(
                "t={}|time={:.1f}|g_loss={:.2f}|d_loss={:.2f}".format(
                    t, t1 - t0, g_loss.numpy(), d_loss.numpy()))
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



