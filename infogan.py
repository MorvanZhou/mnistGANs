# [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/pdf/1606.03657.pdf)

import tensorflow as tf
from tensorflow import keras
import numpy as np
from visual import save_gan
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU
from utils import set_soft_gpu, binary_accuracy, save_weights, class_accuracy
from mnist_ds import get_ds
from gan_cnn import mnist_uni_disc_cnn, mnist_uni_gen_cnn
import time


class InfoGAN(keras.Model):
    """
    discriminator 图片 预测 真假
    q net 图片 预测 c  (c可以理解为 虚拟类别 或 虚拟风格)
    generator z&c 生成 图片
    """
    def __init__(self, rand_dim, style_dim, label_dim, img_shape, fix_std=True, style_scale=2):
        super().__init__()
        self.rand_dim, self.style_dim, self.label_dim = rand_dim, style_dim, label_dim
        self.img_shape = img_shape
        self.fix_std = fix_std
        self.style_scale = style_scale

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")

    def call(self, img_info, training=None, mask=None):
        img_label, img_style = img_info
        noise = tf.random.normal((len(img_label), self.rand_dim))
        if isinstance(img_label, np.ndarray):
            img_label = tf.convert_to_tensor(img_label, dtype=tf.int32)
        if isinstance(img_style, np.ndarray):
            img_style = tf.convert_to_tensor(img_style, dtype=tf.float32)
        return self.g.call([noise, img_label, img_style], training=training)

    def _get_discriminator(self):
        img = Input(shape=self.img_shape)
        s = Sequential([
            mnist_uni_disc_cnn(self.img_shape),
            Dense(1024)
        ])
        style_dim = self.style_dim if self.fix_std else self.style_dim * 2
        q = Sequential([
            Dense(128, input_shape=(1024,)),
            BatchNormalization(),
            LeakyReLU(),
            Dense(style_dim+self.label_dim)
        ], name="recognition")
        o = s(img)
        o_bool = Dense(1)(o)
        o_q = q(o)
        if self.fix_std:
            q_style = self.style_scale*tf.tanh(o_q[:, :style_dim])
        else:
            q_style = tf.concat(
                (self.style_scale * tf.tanh(o_q[:, :style_dim//2]), tf.nn.relu(o_q[:, style_dim//2:style_dim])),
                axis=1)
        q_label = o_q[:, style_dim:]
        model = Model(img, [o_bool, q_style, q_label], name="discriminator")
        print(model.summary())
        return model

    def _get_generator(self):
        latent_dim = self.rand_dim + self.label_dim + self.style_dim
        noise = Input(shape=(self.rand_dim,))
        style = Input(shape=(self.style_dim, ))
        label = Input(shape=(), dtype=tf.int32)
        label_onehot = tf.one_hot(label, depth=self.label_dim)
        model_in = tf.concat((noise, label_onehot, style), axis=1)
        s = mnist_uni_gen_cnn((latent_dim,))
        o = s(model_in)
        model = Model([noise, label, style], o, name="generator")
        print(model.summary())
        return model

    def loss_mutual_info(self, style, pred_style, label, pred_label):
        categorical_loss = keras.losses.sparse_categorical_crossentropy(label, pred_label, from_logits=True)
        if self.fix_std:
            style_mean = pred_style
            style_std = tf.ones_like(pred_style)   # fixed std
        else:
            split = pred_style.shape[1]//2
            style_mean, style_std = pred_style[:split], pred_style[split:]
            style_std = tf.sqrt(tf.exp(style_std))
        epsilon = (style - style_mean) / (style_std + 1e-5)
        ll_continuous = tf.reduce_sum(
            - 0.5 * tf.math.log(2 * np.pi) - tf.math.log(style_std + 1e-5) - 0.5 * tf.square(epsilon),
            axis=1,
        )
        loss = categorical_loss - ll_continuous
        return loss

    def train_d(self, real_fake_img, real_fake_d_label, fake_img_label, fake_style):
        with tf.GradientTape() as tape:
            pred_bool, pred_style, pred_class = self.d.call(real_fake_img, training=True)
            info_split = len(real_fake_d_label)
            real_fake_pred_bool = pred_bool[:info_split]
            loss_bool = self.loss_bool(real_fake_d_label, real_fake_pred_bool)
            fake_pred_style = pred_style[-info_split:]
            fake_pred_label = pred_class[-info_split:]
            loss_info = self.loss_mutual_info(fake_style, fake_pred_style, fake_img_label, fake_pred_label)
            loss = tf.reduce_mean(loss_bool + LAMBDA * loss_info)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss, binary_accuracy(real_fake_d_label, real_fake_pred_bool), class_accuracy(fake_img_label, fake_pred_label)

    def train_g(self, random_img_label, random_img_style):
        d_label = tf.ones((len(random_img_label), 1), tf.float32)   # let d think generated images are real
        with tf.GradientTape() as tape:
            g_img = self.call([random_img_label, random_img_style], training=True)
            pred_bool, pred_style, pred_class = self.d.call(g_img, training=False)
            loss_bool = self.loss_bool(d_label, pred_bool)
            loss_info = self.loss_mutual_info(random_img_style, pred_style, random_img_label, pred_class)
            loss = tf.reduce_mean(loss_bool + LAMBDA * loss_info)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img, binary_accuracy(d_label, pred_bool)

    def step(self, real_img, real_img_label):
        random_img_label = tf.convert_to_tensor(np.random.randint(0, 10, len(real_img)*2), dtype=tf.int32)
        random_img_style = tf.random.uniform((len(real_img)*2, self.style_dim), -self.style_scale, self.style_scale)
        g_loss, g_img, g_bool_acc = self.train_g(random_img_label, random_img_style)

        real_fake_img = tf.concat((real_img, g_img), axis=0)    # 32+64
        real_fake_d_label = tf.concat(      # 32+32
            (tf.ones((len(real_img_label), 1), tf.float32), tf.zeros((len(g_img)//2, 1), tf.float32)), axis=0)
        d_loss, d_bool_acc, d_class_acc = self.train_d(real_fake_img, real_fake_d_label, random_img_label, random_img_style)
        return g_img, d_loss, d_bool_acc, g_loss, g_bool_acc, random_img_label, d_class_acc


def train():
    ds = get_ds(BATCH_SIZE)
    gan = InfoGAN(RAND_DIM, STYLE_DIM, LABEL_DIM, IMG_SHAPE, FIX_STD, STYLE_SCALE)
    t0 = time.time()
    for ep in range(EPOCH):
        for t, (real_img, real_img_label) in enumerate(ds):
            g_img, d_loss, d_bool_acc, g_loss, g_bool_acc, g_img_label, d_class_acc = gan.step(real_img, real_img_label)
            if t % 400 == 0:
                t1 = time.time()
                print("ep={} | time={:.1f}|t={}|d_acc={:.2f}|d_classacc={:.2f}|g_acc={:.2f}|d_loss={:.2f}|g_loss={:.2f}".format(
                    ep, t1-t0, t, d_bool_acc.numpy(), g_bool_acc.numpy(), d_class_acc.numpy(), d_loss.numpy(), g_loss.numpy(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)


if __name__ == "__main__":
    STYLE_DIM = 2
    LABEL_DIM = 10
    RAND_DIM = 64
    LAMBDA = 1
    IMG_SHAPE = (28, 28, 1)
    FIX_STD = True
    STYLE_SCALE = 1
    BATCH_SIZE = 64
    EPOCH = 30

    set_soft_gpu(True)
    train()