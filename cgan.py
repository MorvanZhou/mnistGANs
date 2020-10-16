# [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

import tensorflow as tf
from tensorflow import keras
import numpy as np
from visual import save_gan
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Input, Embedding
from utils import set_soft_gpu, binary_accuracy, save_weights
from mnist_ds import get_ds
from gan_cnn import mnist_uni_disc_cnn, mnist_uni_gen_cnn
import time


class CGAN(keras.Model):
    """
    discriminator 标签+图片 预测 真假
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
        self.loss_func = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, target_labels, training=None, mask=None):
        noise = tf.random.normal((len(target_labels), self.latent_dim))
        if isinstance(target_labels, np.ndarray):
            target_labels = tf.convert_to_tensor(target_labels, dtype=tf.int32)
        return self.g.call([noise, target_labels], training=training)

    def _get_discriminator(self):
        img = Input(shape=self.img_shape)
        label = Input(shape=(), dtype=tf.int32)
        label_emb = Embedding(10, 32)(label)
        emb_img = Reshape((28, 28, 1))(Dense(28*28, activation=keras.activations.relu)(label_emb))
        concat_img = tf.concat((img, emb_img), axis=3)
        s = Sequential([
            mnist_uni_disc_cnn(input_shape=[28, 28, 2]),
            Dense(1)
        ])
        o = s(concat_img)
        model = Model([img, label], o, name="discriminator")
        print(model.summary())
        return model

    def _get_generator(self):
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(), dtype=tf.int32)
        label_onehot = tf.one_hot(label, depth=self.label_dim)
        model_in = tf.concat((noise, label_onehot), axis=1)
        s = mnist_uni_gen_cnn((self.latent_dim+self.label_dim,))
        o = s(model_in)
        model = Model([noise, label], o, name="generator")
        print(model.summary())
        return model

    def train_d(self, img, img_label, label):
        with tf.GradientTape() as tape:
            pred = self.d.call([img, img_label], training=True)
            loss = self.loss_func(label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss, binary_accuracy(label, pred)

    def train_g(self, random_img_label):
        d_label = tf.ones((len(random_img_label), 1), tf.float32)   # let d think generated images are real
        with tf.GradientTape() as tape:
            g_img = self.call(random_img_label, training=True)
            pred = self.d.call([g_img, random_img_label], training=False)
            loss = self.loss_func(d_label, pred)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img, binary_accuracy(d_label, pred)

    def step(self, real_img, real_img_label):
        random_img_label = tf.convert_to_tensor(np.random.randint(0, 10, len(real_img)*2), dtype=tf.int32)
        g_loss, g_img, g_acc = self.train_g(random_img_label)

        img = tf.concat((real_img, g_img[:len(g_img)//2]), axis=0)
        img_label = tf.concat((real_img_label, random_img_label[:len(g_img)//2]), axis=0)
        d_label = tf.concat((tf.ones((len(real_img_label), 1), tf.float32), tf.zeros((len(g_img)//2, 1), tf.float32)), axis=0)
        d_loss, d_acc = self.train_d(img, img_label, d_label)
        return g_img, d_loss, d_acc, g_loss, g_acc, random_img_label


def train():
    ds = get_ds(BATCH_SIZE)
    gan = CGAN(LATENT_DIM, LABEL_DIM, IMG_SHAPE)
    t0 = time.time()
    for ep in range(EPOCH):
        for t, (real_img, real_img_label) in enumerate(ds):
            g_img, d_loss, d_acc, g_loss, g_acc, g_img_label = gan.step(real_img, real_img_label)
            if t % 400 == 0:
                t1 = time.time()
                print("ep={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".format(
                    ep, t1-t0, t, d_acc.numpy(), g_acc.numpy(), d_loss.numpy(), g_loss.numpy(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)


if __name__ == "__main__":
    LATENT_DIM = 100
    IMG_SHAPE = (28, 28, 1)
    LABEL_DIM = 10
    BATCH_SIZE = 64
    EPOCH = 20

    set_soft_gpu(True)
    train()






