# [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf)
import tensorflow as tf
from utils import set_soft_gpu
from wgan import WGAN, train
from mnist_ds import get_train_x


class WGANgp(WGAN):
    """
    WGAN clip weights 方案比较粗暴,
    用 gradient penalty 替代 clip 有助于 D 的能力提升, 间接提升 G.
    """
    def __init__(self, latent_dim, lambda_, k, img_shape):
        super().__init__(latent_dim, None, img_shape)
        self.lambda_ = lambda_
        self.k = k
        self.d = self._get_discriminator(use_bn=False)      # no critic batch norm
        self.opt = tf.keras.optimizers.Adam(0.0002, beta_1=0, beta_2=0.9)

    # gradient penalty
    def gp(self, real_img, fake_img):
        e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
        noise_img = e * real_img + (1.-e)*fake_img      # extend distribution space
        with tf.GradientTape() as tape:
            tape.watch(noise_img)
            o = self.d.call(noise_img, training=True)
        g = tape.gradient(o, noise_img)         # image gradients
        g_norm2 = tf.sqrt(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))  # norm2 penalty
        gp = tf.square(g_norm2 - self.k)
        return gp

    def train_d(self, img):
        g_img = self.call(len(img), training=False)
        gp = self.gp(img, g_img)
        all_img = tf.concat((img, g_img), axis=0)
        with tf.GradientTape() as tape:
            pred = self.d.call(all_img, training=True)
            pred_true, pred_fake = pred[:len(img)], pred[len(img):]
            w_loss = -tf.reduce_mean(pred_true - pred_fake)  # maximize W distance
            gp_loss = self.lambda_ * gp
        grads = tape.gradient(w_loss+gp_loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return w_loss


if __name__ == "__main__":
    LATENT_DIM = 100
    D_LOOP = 5
    LAMBDA = 10
    K = 1
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 64
    STEP = 20001

    set_soft_gpu(True)
    d = get_train_x()
    m = WGANgp(LATENT_DIM, LAMBDA, K, IMG_SHAPE)
    train(m, d, STEP, D_LOOP, BATCH_SIZE)


