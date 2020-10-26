# [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)

import tensorflow as tf
from tensorflow import keras
import numpy as np
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, save_weights
from mnist_ds import get_train_x, get_test_x, downsampling
import time


class SRGAN(keras.Model):
    """
    低清图片高清化
    感知相似性(对抗损失+内容损失)生成逼真高清图片.
    原算法中使用预训练 VGG 作为 feature map 的提取器, 对计算"生成像素误差", 我不想引入太多网络,
    所以我直接使用 Discriminator 的 feature map 做这件事.
    """
    def __init__(self, lr_img_shape, hr_img_shape, lambda_adver):
        super().__init__()
        self.lr_img_shape = lr_img_shape
        self.hr_img_shape = hr_img_shape
        self.lambda_adver = lambda_adver

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_mse = keras.losses.MeanSquaredError()

    def call(self, img, training=None, mask=None):
        if isinstance(img, np.ndarray):
            img = tf.convert_to_tensor(img, dtype=tf.float32)
        return self.g.call(img, training=training)

    def _get_discriminator(self):
        x = keras.Input(self.hr_img_shape)
        s1 = keras.Sequential([
            keras.layers.Conv2D(64, 5, strides=2, padding='same', input_shape=self.hr_img_shape),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.3),
            keras.layers.Conv2D(128, 5, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
        ])
        s2 = keras.Sequential([
            keras.layers.Dropout(0.3),
            keras.layers.Flatten(),
            keras.layers.Dense(1),
        ])
        # don't have pretrained VGG for extracting feature map
        # try use feature map from discriminator instead
        feature_map = s1(x)
        o = s2(feature_map)
        model = keras.Model(x, [o, feature_map])
        model.summary()
        return model

    def _get_generator(self):
        pre_process = keras.Sequential([
            # -> [n, 7, 7, 64]
            keras.layers.Conv2D(64, 3, strides=1, padding='same', input_shape=self.lr_img_shape),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
        ], name="pre_process")
        encoder = keras.Sequential([
            # -> [n, 7, 7, 64]
            keras.layers.Conv2D(64, 3, strides=1, padding='same', input_shape=(7, 7, 64)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(64, 3, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
        ], name="encoder")

        decoder = keras.Sequential([
            # -> [n, 14, 14, 64]
            keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', input_shape=(7, 7, 64)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            # -> [n, 28, 28, 32]
            keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            # -> [n, 28, 28, 1]
            keras.layers.Conv2D(self.hr_img_shape[-1], 3, padding='same', activation=keras.activations.tanh)
        ], name="decoder")

        x = keras.Input(self.lr_img_shape)
        _x = pre_process(x)
        z = encoder(_x)
        o = decoder(z + _x)
        model = keras.Model(x, o, name="generator")
        model.summary()
        return model

    def train_d(self, img, label):
        with tf.GradientTape() as tape:
            pred, _ = self.d.call(img, training=True)
            loss = self.loss_bool(label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss

    def train_g(self, lr_img, hr_img):
        d_label = tf.ones((len(lr_img), 1), tf.float32)   # let d think generated images are real
        with tf.GradientTape() as tape:
            sr_img = self.g.call(lr_img, training=True)
            # don't have pretrained VGG for extracting feature map, try disc's feature map instead
            pred, sr_feature_map = self.d.call(sr_img, training=False)
            _, hr_feature_map = self.d.call(hr_img, training=False)
            loss_adver = self.lambda_adver * self.loss_bool(d_label, pred)
            loss_content = self.loss_mse(hr_feature_map, sr_feature_map)
            loss = loss_content + loss_adver
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, sr_img

    def step(self, lr_img, hr_img):
        g_loss, sr_img = self.train_g(lr_img, hr_img)

        half = len(sr_img)//2
        img = tf.concat((hr_img[:half], sr_img[half:]), axis=0)
        # patched label
        d_label = tf.concat(
            (tf.ones((half, 1), tf.float32), tf.zeros((half, 1), tf.float32)), axis=0)
        d_loss = self.train_d(img, d_label)
        return d_loss, g_loss


def train(gan, hr, lr_test, steps, batch_size):
    t0 = time.time()
    for t in range(steps):
        idx = np.random.randint(0, len(hr), batch_size)
        hr_img = tf.gather(hr, idx)
        lr_img = downsampling(hr_img, gan.lr_img_shape)
        g_loss, d_loss = gan.step(lr_img, hr_img)
        if t % 500 == 0:
            save_gan(gan, t, img=lr_test)
            t1 = time.time()
            print(
                "t={}|time={:.1f}|g_loss={:.2f}|d_loss={:.2f}".format(
                    t, t1 - t0, g_loss.numpy(), d_loss.numpy()))
            t0 = t1
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    LR_IMG_SHAPE = (7, 7, 1)
    HR_IMG_SHAPE = (28, 28, 1)
    LAMBDA_ADVER = 5e-2
    BATCH_SIZE = 64
    STEPS = 10001

    set_soft_gpu(True)
    HR = get_train_x()
    LR_TEST = downsampling(get_test_x(), to_shape=LR_IMG_SHAPE)
    m = SRGAN(LR_IMG_SHAPE, HR_IMG_SHAPE, LAMBDA_ADVER)
    train(m, HR, LR_TEST, STEPS, BATCH_SIZE)






