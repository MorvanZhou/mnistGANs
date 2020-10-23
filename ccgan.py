# [Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1611.06430.pdf)

import tensorflow as tf
from tensorflow import keras
import numpy as np
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, binary_accuracy, save_weights
from mnist_ds import get_ds, get_test_x
from gan_cnn import mnist_uni_disc_cnn, mnist_uni_img2img
import time


class CCGAN(keras.Model):
    """
    生成图片中被遮挡的部分
    """
    def __init__(self, label_dim, mask_range, img_shape):
        super().__init__()
        self.label_dim = label_dim
        self.mask_range = mask_range
        self.img_shape = img_shape

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_class = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, img, training=None, mask=None):
        if isinstance(img, np.ndarray):
            img = tf.convert_to_tensor(img, dtype=tf.float32)
        return self.g.call(img, training=training)

    def _get_discriminator(self):
        img = keras.Input(shape=self.img_shape)
        s = keras.Sequential([
            keras.layers.GaussianNoise(0.01, input_shape=self.img_shape),   # add some noise
            mnist_uni_disc_cnn(),
            keras.layers.Dense(1+self.label_dim)
        ])
        o = s(img)
        o_bool, o_class = o[:, :-self.label_dim], o[:, -self.label_dim:]
        model = keras.Model(img, [o_bool, o_class], name="discriminator")
        model.summary()
        return model

    def _get_generator(self):
        model = mnist_uni_img2img(self.img_shape)
        model.summary()
        return model

    def train_d(self, img, img_label, label):
        with tf.GradientTape() as tape:
            pred_bool, pred_class = self.d.call([img, img_label], training=True)
            loss = self.loss_bool(label, pred_bool) + self.loss_class(img_label, pred_class)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss, binary_accuracy(label, pred_bool)

    def train_g(self, img, img_label):
        masked_img = self.get_rand_masked(img)
        d_label = tf.ones((len(img_label), 1), tf.float32)   # let d think generated images are real
        with tf.GradientTape() as tape:
            g_img = self.g.call(masked_img, training=True)
            pred_bool, pred_class = self.d.call([g_img, img_label], training=False)
            loss = self.loss_bool(d_label, pred_bool) + self.loss_class(img_label, pred_class)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img, binary_accuracy(d_label, pred_bool)

    def get_rand_masked(self, img):
        mask_width = np.random.randint(self.mask_range[0], self.mask_range[1])
        mask_xy = np.random.randint(0, self.img_shape[0] - mask_width, (2,))
        mask = np.ones(self.img_shape, np.float32)
        mask[mask_xy[0]:mask_width + mask_xy[0], mask_xy[0]:mask_width + mask_xy[0]] = 0
        mask = tf.convert_to_tensor(np.expand_dims(mask, axis=0))
        masked_img = img * mask
        return masked_img

    def step(self, real_img, real_img_label):
        g_loss, g_img, g_acc = self.train_g(real_img, real_img_label)

        half = len(g_img)//2
        img = tf.concat((real_img[:half], g_img[half:]), axis=0)
        d_label = tf.concat((tf.ones((half, 1), tf.float32), tf.zeros((half, 1), tf.float32)), axis=0)
        d_loss, d_acc = self.train_d(img, real_img_label, d_label)
        return d_loss, d_acc, g_loss, g_acc


def train(gan, ds, test_x):
    t0 = time.time()
    for ep in range(EPOCH):
        for t, (real_img, real_img_label) in enumerate(ds):
            d_loss, d_acc, g_loss, g_acc = gan.step(real_img, real_img_label)
            if t % 400 == 0:
                t1 = time.time()
                print("ep={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".format(
                    ep, t1-t0, t, d_acc.numpy(), g_acc.numpy(), d_loss.numpy(), g_loss.numpy(), ))
                t0 = t1
        save_gan(gan, ep, img=test_x)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    LATENT_DIM = 100
    MASK_RANGE = (10, 16)
    IMG_SHAPE = (28, 28, 1)
    LABEL_DIM = 10
    BATCH_SIZE = 64
    EPOCH = 20

    set_soft_gpu(True)
    test_x = get_test_x()
    d = get_ds(BATCH_SIZE)
    m = CCGAN(LATENT_DIM, MASK_RANGE, IMG_SHAPE)
    train(m, d, test_x)






