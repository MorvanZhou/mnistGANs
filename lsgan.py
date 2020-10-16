# [Least Squares Generative Adversarial Networks](https://arxiv.org/pdf/1611.04076.pdf)
import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu
from mnist_ds import get_ds
from gan import GAN, train


class LSGAN(GAN):
    """
    和原始 GAN 相比，只是修改了 loss function, 但有时在最开始的时候崩掉, 生成空白
    """
    def __init__(self, latent_dim, img_shape, a, b, c):
        super().__init__(latent_dim, img_shape)
        self.a, self.b, self.c = a, b, c
        self.loss_func = keras.losses.MeanSquaredError()

    def step(self, img):
        d_label = self.c * tf.ones((len(img) * 2, 1), tf.float32)
        g_loss, g_img, g_acc = self.train_g(d_label)

        d_label = tf.concat(
            (self.b * tf.ones((len(img), 1), tf.float32),       # real
             self.a * tf.ones((len(g_img)//2, 1), tf.float32)), # fake
            axis=0)
        img = tf.concat((img, g_img[:len(g_img)//2]), axis=0)
        d_loss, d_acc = self.train_d(img, d_label)
        return d_loss, d_acc, g_loss, g_acc


if __name__ == "__main__":
    LATENT_DIM = 100
    # A, B, C = -1, 1, 0
    A, B, C = -1, 1, 1
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 64
    EPOCH = 20

    set_soft_gpu(True)
    d = get_ds(BATCH_SIZE)
    m = LSGAN(LATENT_DIM, IMG_SHAPE, A, B, C)
    train(m, d, EPOCH)



