# [Wasserstein Divergence for GANs](https://arxiv.org/pdf/1712.01026.pdf)
import tensorflow as tf
from utils import set_soft_gpu, save_weights
from wgan import WGAN
from mnist_ds import get_train_x
import time
import numpy as np
from visual import save_gan, cvt_gif


class WGANdiv(WGAN):
    """
    WGAN clip weights 方案比较粗暴,
    用 gradient penalty 替代 clip 有助于 D 的能力提升, 间接提升 G. 不过WGAN-GP中判别器的优化目标已经不再是散度
    提出了Wasserstein散度的概念，摆脱了WGAN需要 [公式] 满足Lipschitz条件的限制。
    """
    def __init__(self, latent_dim, p, k, img_shape):
        super().__init__(latent_dim, None, img_shape)
        self.p = p
        self.k = k
        self.opt = tf.keras.optimizers.Adam(0.0002, beta_1=0, beta_2=0.9)

    # Wasserstein Divergence
    def gp(self, real_img, fake_img):
        e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
        noise_img = e * real_img + (1.-e)*fake_img      # extend distribution space
        with tf.GradientTape() as tape:
            tape.watch(noise_img)
            o = self.d.call(noise_img, training=True)
        g = tape.gradient(o, noise_img)         # image gradients
        gp = tf.pow(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]), self.p)
        return tf.reduce_mean(gp)

    def train_d(self, img):
        g_img = self.call(len(img), training=False)
        gp = self.gp(img, g_img)
        all_img = tf.concat((img, g_img), axis=0)
        with tf.GradientTape() as tape:
            pred = self.d.call(all_img, training=True)
            pred_true, pred_fake = pred[:len(img)], pred[len(img):]
            w_loss = -tf.reduce_mean(pred_true - pred_fake)  # maximize W distance
            gp_loss = self.k * gp
            loss = w_loss+gp_loss
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return w_loss


def train(gan, ds, steps, batch_size):
    t0 = time.time()
    for t in range(steps):
        idx = np.random.randint(0, len(ds), batch_size)
        img = tf.gather(ds, idx)
        d_loss = gan.train_d(img)
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
    D_LOOP = 5
    P = 6
    K = 2
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 64
    STEP = 20001

    set_soft_gpu(True)
    d = get_train_x()
    m = WGANdiv(LATENT_DIM, P, K, IMG_SHAPE)
    train(m, d, STEP, BATCH_SIZE)


