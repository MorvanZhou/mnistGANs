# [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf)
import tensorflow as tf
from utils import set_soft_gpu
from wgan import WGAN, train
from mnist_ds import get_train_x


# modified from WGAN
class WGANgp(WGAN):
    """
    WGAN clip weights 方案比较粗暴,
    用 gradient penalty 替代 clip 有助于 D 的能力提升, 间接提升 G.
    """
    def __init__(self, latent_dim, lambda_, img_shape):
        super().__init__(latent_dim, None, img_shape)
        self.lambda_ = lambda_

    def _build_d(self):
        self.d = self._get_discriminator(use_bn=False)      # no critic batch norm

    # gradient penalty
    def gp(self, real_img, fake_img):
        e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
        noise_img = e * real_img + (1.-e)*fake_img      # extend distribution space
        with tf.GradientTape() as tape:
            tape.watch(noise_img)
            o = self.d(noise_img)
        g = tape.gradient(o, noise_img)         # image gradients
        g_norm2 = tf.sqrt(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))  # norm2 penalty
        gp = tf.square(g_norm2 - 1.)
        return tf.reduce_mean(gp)

    def train_d(self, real_img):
        with tf.GradientTape() as tape:
            fake_img = self.call(len(real_img), training=False)
            gp = self.gp(real_img, fake_img)
            pred_real = self.d.call(real_img, training=True)
            pred_fake = self.d.call(fake_img, training=True)
            w_loss = self.w_distance(pred_fake, pred_real)  # maximize W distance
            loss = w_loss + self.lambda_ * gp       # add gradient penalty
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return w_loss


if __name__ == "__main__":
    LATENT_DIM = 100
    D_LOOP = 5
    LAMBDA = 10
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 64
    STEP = 20001

    set_soft_gpu(True)
    d = get_train_x()
    m = WGANgp(LATENT_DIM, LAMBDA, IMG_SHAPE)
    train(m, d, STEP, D_LOOP, BATCH_SIZE)


