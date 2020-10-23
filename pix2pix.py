# [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)

import tensorflow as tf
from tensorflow import keras
import numpy as np
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, save_weights
from mnist_ds import get_ds, get_test_x
from gan_cnn import mnist_uni_disc_cnn, mnist_unet
import time


class Pix2Pix(keras.Model):
    """
    根据输入图片,按要求生成输出图片
    """
    def __init__(self, mask_range, img_shape, lambda_):
        super().__init__()
        self.mask_range = mask_range
        self.img_shape = img_shape
        self.lambda_ = lambda_

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_l1 = keras.losses.MeanAbsoluteError()

    def call(self, img, training=None, mask=None):
        if isinstance(img, np.ndarray):
            img = tf.convert_to_tensor(img, dtype=tf.float32)
        return self.g.call(img, training=training)

    def _get_discriminator(self):
        input_img = keras.Input(shape=self.img_shape)
        generated_img = keras.Input(shape=self.img_shape)
        concat_img = tf.concat((input_img, generated_img), axis=-1)
        # -> [n, 7, 7, 128]
        s = keras.Sequential([
            *mnist_uni_disc_cnn(
                input_shape=[self.img_shape[0], self.img_shape[1], self.img_shape[2]*2]
            ).layers[:-1],  # remove flatten
            keras.layers.Conv2D(1, (4, 4))
        ])
        o = tf.squeeze(s(concat_img), axis=-1)
        # [patch gan img](https://www.researchgate.net/profile/Gozde_Unal4/publication/323904616/figure/fig1/AS:606457334595585@1521602104652/PatchGAN-discriminator-Each-value-of-the-output-matrix-represents-the-probability-of.png)
        patch_gan = keras.Model([input_img, generated_img], o, name="patch_gan")
        patch_gan.summary()
        return patch_gan

    def _get_generator(self):
        model = mnist_unet(self.img_shape)
        model.summary()
        return model

    def train_d(self, input_img, img, label):
        with tf.GradientTape() as tape:
            pred = self.d.call([input_img, img], training=True)
            loss = self.loss_bool(label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss

    def train_g(self, input_img, real_img):
        # patched label
        d_label = tf.ones((len(real_img), 4, 4), tf.float32)   # let d think generated images are real
        with tf.GradientTape() as tape:
            g_img = self.g.call(input_img, training=True)
            pred = self.d.call([input_img, g_img], training=False)
            loss = self.loss_bool(d_label, pred) + self.lambda_ * self.loss_l1(real_img, g_img)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img

    def get_rand_masked(self, img):
        mask_width = np.random.randint(self.mask_range[0], self.mask_range[1])
        mask_xy = np.random.randint(0, self.img_shape[0] - mask_width, (2,))
        mask = np.ones(self.img_shape, np.float32)
        mask[mask_xy[0]:mask_width + mask_xy[0], mask_xy[0]:mask_width + mask_xy[0]] = 0
        mask = tf.convert_to_tensor(np.expand_dims(mask, axis=0))
        masked_img = img * mask
        return masked_img

    def step(self, real_img):
        input_img = self.get_rand_masked(real_img)
        g_loss, g_img = self.train_g(input_img, real_img)

        half = len(g_img)//2
        img = tf.concat((real_img[:half], g_img[half:]), axis=0)
        # patched label
        d_label = tf.concat(
            (tf.ones((half, 4, 4), tf.float32), tf.zeros((half, 4, 4), tf.float32)), axis=0)
        d_loss = self.train_d(input_img, img, d_label)
        return d_loss, g_loss


def train(gan, ds, test_x):
    t0 = time.time()
    for ep in range(EPOCH):
        for t, (real_img, _) in enumerate(ds):
            d_loss, g_loss = gan.step(real_img)
            if t % 400 == 0:
                t1 = time.time()
                print("ep={} | time={:.1f} | t={} | d_loss={:.2f} | g_loss={:.2f}".format(
                    ep, t1-t0, t, d_loss.numpy(), g_loss.numpy()))
                t0 = t1
        save_gan(gan, ep, img=test_x)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    MASK_RANGE = (10, 16)
    IMG_SHAPE = (28, 28, 1)
    LAMBDA = 1
    BATCH_SIZE = 64
    EPOCH = 20

    set_soft_gpu(True)
    test_x = get_test_x()
    d = get_ds(BATCH_SIZE)
    m = Pix2Pix(MASK_RANGE, IMG_SHAPE, LAMBDA)
    train(m, d, test_x)






