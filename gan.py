# [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
# [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from utils import set_soft_gpu, binary_accuracy, save_weights
from mnist_ds import get_half_batch_ds
from gan_cnn import mnist_uni_disc_cnn, mnist_uni_gen_cnn
import time


class GAN(keras.Model):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_func = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, n, training=None):
        return self.g.call(tf.random.normal((n, self.latent_dim)), training=training)

    def _get_generator(self):
        model = mnist_uni_gen_cnn((self.latent_dim,))
        model.summary()
        return model

    def _get_discriminator(self):
        model = Sequential([
            mnist_uni_disc_cnn(self.img_shape),
            Dense(1)
        ], name="discriminator")
        model.summary()
        return model

    def train_d(self, img, label):
        with tf.GradientTape() as tape:
            pred = self.d.call(img, training=True)
            loss = self.loss_func(label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss, binary_accuracy(label, pred)

    def train_g(self, d_label):
        with tf.GradientTape() as tape:
            g_img = self.call(len(d_label), training=True)
            pred = self.d.call(g_img, training=False)
            loss = self.loss_func(d_label, pred)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img, binary_accuracy(d_label, pred)

    def step(self, img):
        d_label = tf.ones((len(img) * 2, 1), tf.float32)  # let d think generated images are real
        g_loss, g_img, g_acc = self.train_g(d_label)

        d_label = tf.concat((tf.ones((len(img), 1), tf.float32), tf.zeros((len(g_img)//2, 1), tf.float32)), axis=0)
        img = tf.concat((img, g_img[:len(g_img)//2]), axis=0)
        d_loss, d_acc = self.train_d(img, d_label)
        return d_loss, d_acc, g_loss, g_acc


def train(gan, ds, epoch):
    t0 = time.time()
    for ep in range(epoch):
        for t, (img, _) in enumerate(ds):
            d_loss, d_acc, g_loss, g_acc = gan.step(img)
            if t % 400 == 0:
                t1 = time.time()
                print(
                    "ep={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".format(
                        ep, t1 - t0, t, d_acc.numpy(), g_acc.numpy(), d_loss.numpy(), g_loss.numpy(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    LATENT_DIM = 100
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 64
    EPOCH = 20

    set_soft_gpu(True)
    d = get_half_batch_ds(BATCH_SIZE)
    m = GAN(LATENT_DIM, IMG_SHAPE)
    train(m, d, EPOCH)



