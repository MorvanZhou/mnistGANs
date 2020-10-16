# [Conditional Image Synthesis with Auxiliary Classifier GANs](https://arxiv.org/pdf/1610.09585.pdf)

import tensorflow as tf
from tensorflow import keras
import numpy as np
from visual import save_gan, cvt_gif
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from utils import set_soft_gpu, binary_accuracy, save_weights
from mnist_ds import get_ds
from gan_cnn import mnist_uni_disc_cnn, mnist_uni_gen_cnn
import time


class ACGAN(keras.Model):
    """
    discriminator 图片 预测 真假+标签
    generator 标签 生成 图片
    """
    def __init__(self, latent_dim, label_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.img_shape = img_shape

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
        self.loss_class = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def call(self, target_labels, training=None, mask=None):
        noise = tf.random.normal((len(target_labels), self.latent_dim))
        if isinstance(target_labels, np.ndarray):
            target_labels = tf.convert_to_tensor(target_labels, dtype=tf.int32)
        return self.g.call([noise, target_labels], training=training)

    def _get_discriminator(self):
        img = Input(shape=self.img_shape)
        s = Sequential([
            mnist_uni_disc_cnn(input_shape=self.img_shape),
            Dense(11)
        ])
        o = s(img)
        o_bool, o_class = o[:, :1], o[:, 1:]
        model = Model(img, [o_bool, o_class], name="discriminator")
        model.summary()
        return model

    def _get_generator(self):
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(), dtype=tf.int32)
        label_onehot = tf.one_hot(label, depth=self.label_dim)
        model_in = tf.concat((noise, label_onehot), axis=1)
        s = mnist_uni_gen_cnn((self.latent_dim+self.label_dim,))
        o = s(model_in)
        model = Model([noise, label], o, name="generator")
        model.summary()
        return model

    def train_d(self, img, img_label, label):
        with tf.GradientTape() as tape:
            pred_bool, pred_class = self.d.call(img, training=True)
            loss_bool = self.loss_bool(label, pred_bool)
            loss_class = self.loss_class(img_label, pred_class)
            loss = tf.reduce_mean(loss_bool + loss_class)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss, binary_accuracy(label, pred_bool)

    def train_g(self, random_img_label):
        d_label = tf.ones((len(random_img_label), 1), tf.float32)   # let d think generated images are real
        with tf.GradientTape() as tape:
            g_img = self.call(random_img_label, training=True)
            pred_bool, pred_class = self.d.call(g_img, training=False)
            loss_bool = self.loss_bool(d_label, pred_bool)
            loss_class = self.loss_class(random_img_label, pred_class)
            loss = tf.reduce_mean(loss_bool + loss_class)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img, binary_accuracy(d_label, pred_bool)

    def step(self, real_img, real_img_label):
        random_img_label = tf.convert_to_tensor(np.random.randint(0, 10, len(real_img)*2), dtype=tf.int32)
        g_loss, g_img, g_bool_loss = self.train_g(random_img_label)

        img = tf.concat((real_img, g_img[:len(g_img)//2]), axis=0)
        img_label = tf.concat((real_img_label, random_img_label[:len(g_img) // 2]), axis=0)
        d_label = tf.concat((tf.ones((len(real_img_label), 1), tf.float32), tf.zeros((len(g_img)//2, 1), tf.float32)), axis=0)
        d_loss, d_bool_acc = self.train_d(img, img_label, d_label)
        return g_img, d_loss, d_bool_acc, g_loss, g_bool_loss, random_img_label


def train(gan, ds):
    t0 = time.time()
    for ep in range(EPOCH):
        for t, (real_img, real_img_label) in enumerate(ds):
            g_img, d_loss, d_bool_acc, g_loss, g_bool_loss, g_img_label = gan.step(real_img, real_img_label)
            if t % 400 == 0:
                t1 = time.time()
                print("ep={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".format(
                    ep, t1-t0, t, d_bool_acc.numpy(), g_bool_loss.numpy(), d_loss.numpy(), g_loss.numpy(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    LATENT_DIM = 100
    IMG_SHAPE = (28, 28, 1)
    LABEL_DIM = 10
    BATCH_SIZE = 64
    EPOCH = 20

    set_soft_gpu(True)
    d = get_ds(BATCH_SIZE)
    m = ACGAN(LATENT_DIM, LABEL_DIM, IMG_SHAPE)
    train(m, d)