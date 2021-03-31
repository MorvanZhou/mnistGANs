# [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
import numpy as np
from utils import set_soft_gpu, save_weights
from mnist_ds import get_train_x
from gan_cnn import mnist_uni_gen_cnn, mnist_uni_disc_cnn
import time


class WGAN(keras.Model):
    """
    Wasserstein 距离作为损失函数， 避免D太强导致G的梯度消失。
    D 最大化 Wasserstein 距离，提高收敛性
    G 最小化 Wasserstein 距离
    Clip D weights，局限住太强的 D，让 G 可以跟上 (Lipschitz 约束)。
    """
    def __init__(self, latent_dim, clip, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.clip = clip
        self.img_shape = img_shape
        self.opt = keras.optimizers.Adam(0.0002, beta_1=0, beta_2=0.9)
        self.g = self._get_generator()
        self._build_d()

    def _build_d(self):
        self.d = self._get_discriminator()

    def call(self, n, training=None, mask=None):
        return self.g.call(tf.random.normal((n, self.latent_dim)), training=training)

    def _get_generator(self):
        model = mnist_uni_gen_cnn((self.latent_dim,))
        model.summary()
        return model

    def _get_discriminator(self, use_bn=True):
        model = keras.Sequential([
            mnist_uni_disc_cnn(self.img_shape, use_bn),
            keras.layers.Dense(1)
        ], name="critic")
        model.summary()
        return model

    @staticmethod
    def w_distance(fake, real=None):
        # the distance of two data distributions
        if real is None:
            return tf.reduce_mean(fake)
        else:
            return tf.reduce_mean(fake) - tf.reduce_mean(real)

    def train_d(self, real_img):
        with tf.GradientTape() as tape:
            fake_img = self.call(len(real_img), training=False)
            pred_real = self.d.call(real_img, training=True)
            pred_fake = self.d.call(fake_img, training=True)
            loss = self.w_distance(pred_fake, pred_real)   # maximize W distance
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        # clip discriminator's weights
        for w in self.d.trainable_weights:
            w.assign(tf.clip_by_value(w, -self.clip, self.clip))
        return loss

    def train_g(self, n):
        with tf.GradientTape() as tape:
            g_img = self.call(n, training=True)
            pred_fake = self.d.call(g_img, training=False)
            loss = -self.w_distance(pred_fake)       # minimize W distance
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss


def train(gan, ds, steps, d_loop, batch_size):
    t0 = time.time()
    for t in range(steps):
        for _ in range(d_loop):
            idx = np.random.randint(0, len(ds), batch_size)
            real_img = tf.gather(ds, idx)
            d_loss = gan.train_d(real_img)
        g_loss = gan.train_g(batch_size)
        if t % 1000 == 0:
            t1 = time.time()
            print("t={} | time={:.1f} | d_loss={:.2f} | g_loss={:.2f}".format(
                    t, t1 - t0, d_loss.numpy(), g_loss.numpy(), ))
            t0 = t1
            save_gan(gan, t)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    LATENT_DIM = 100
    CLIP = 0.01
    D_LOOP = 5
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 64
    STEP = 20001

    set_soft_gpu(True)
    d = get_train_x()
    m = WGAN(LATENT_DIM, CLIP, IMG_SHAPE)
    train(m, d, STEP, D_LOOP, BATCH_SIZE)



