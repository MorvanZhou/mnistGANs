import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

os.makedirs("visual", exist_ok=True)


def show_mnist(n=20):
    from tensorflow import keras
    (x, y), _ = keras.datasets.mnist.load_data()
    idx = np.random.randint(0, len(x), n)
    x, y = x[idx], y[idx]
    n_col = 5
    n_row = len(x) // n_col
    plt.figure(0, (5, n_row))
    for c in range(n_col):
        for r in range(n_row):
            i = r*n_col+c
            plt.subplot(n_row, n_col, i+1)
            plt.imshow(x[i], cmap="gray_r")
            plt.axis("off")
            # plt.xlabel(y[i])
    plt.tight_layout()
    plt.savefig("visual/mnist.png")
    # plt.show()


def save_gan(model, ep):
    name = model.__class__.__name__.lower()
    if name in ["gan", "wgan", "wgangp", "lsgan"]:
        imgs = model.call(100, training=False).numpy()
        _save_gan(name, ep, imgs, show_label=False)
    elif name == "cgan" or name == "acgan":
        img_label = np.arange(0, 10).astype(np.int32).repeat(10, axis=0)
        imgs = model.predict(img_label)
        _save_gan(name, ep, imgs, show_label=True)
    elif name == "infogan":
        img_label = np.arange(0, 10).astype(np.int32).repeat(10, axis=0)
        img_style = np.concatenate(
            [np.linspace(-model.style_scale, model.style_scale, 10)] * 10).reshape((100, 1)).repeat(model.style_dim, axis=1).astype(np.float32)
        img_info = img_label, img_style
        imgs = model.predict(img_info)
        _save_gan(name, ep, imgs, show_label=False)
    else:
        raise ValueError(name)


def _save_gan(model_name, ep, imgs, show_label=False):
    imgs = (imgs + 1) * 255 / 2
    plt.clf()
    nc, nr = 10, 10
    plt.figure(0, (nc * 2, nr * 2))
    for c in range(nc):
        for r in range(nr):
            i = r * nc + c
            plt.subplot(nc, nr, i + 1)
            plt.imshow(imgs[i], cmap="gray_r")
            plt.axis("off")
            if show_label:
                plt.text(23, 26, int(r), fontsize=23)
    plt.tight_layout()
    dir_ = "visual/{}".format(model_name)
    os.makedirs(dir_, exist_ok=True)
    path = dir_ + "/{}.png".format(ep)
    plt.savefig(path)


def infogan_comp():
    import tensorflow as tf
    from infogan import InfoGAN
    STYLE_DIM = 2
    LABEL_DIM = 10
    RAND_DIM = 88
    IMG_SHAPE = (28, 28, 1)
    FIX_STD = True
    model = InfoGAN(RAND_DIM, STYLE_DIM, LABEL_DIM, IMG_SHAPE, FIX_STD)
    model.load_weights("./models/infogan/model.ckpt").expect_partial()
    img_label = np.arange(0, 10).astype(np.int32).repeat(10, axis=0)
    noise = tf.repeat(tf.random.normal((1, model.rand_dim)), len(img_label), axis=0)

    def plot(noise, img_label, img_style, n):
        img_label = tf.convert_to_tensor(img_label, dtype=tf.int32)
        img_style = tf.convert_to_tensor(img_style, dtype=tf.float32)
        imgs = model.g.call([noise, img_label, img_style], training=False).numpy()
        plt.clf()
        nc, nr = 10, 10
        plt.figure(0, (nc * 2, nr * 2))
        for c in range(nc):
            for r in range(nr):
                i = r * nc + c
                plt.subplot(nc, nr, i + 1)
                plt.imshow(imgs[i], cmap="gray_r")
                plt.axis("off")
                plt.text(23, 26, int(r), fontsize=23)
        plt.tight_layout()
        model_name = model.__class__.__name__.lower()
        dir_ = "visual/{}".format(model_name)
        os.makedirs(dir_, exist_ok=True)
        path = dir_ + "/style{}.png".format(n)
        plt.savefig(path)

    img_style = np.concatenate(
        [np.linspace(-model.style_scale, model.style_scale, 10)] * 10).reshape((100, 1)).astype(np.float32)
    plot(noise, img_label, np.concatenate((img_style, np.zeros_like(img_style)), axis=1), 1)
    plot(noise, img_label, np.concatenate((np.zeros_like(img_style), img_style), axis=1), 2)


def cvt_gif(folders_or_gan):
    if not isinstance(folders_or_gan, list):
        folders_or_gan = [folders_or_gan.__class__.__name__.lower()]
    for folder in folders_or_gan:
        folder = "visual/"+folder
        fs = [folder+"/" + f for f in os.listdir(folder)]
        imgs = []
        for f in sorted(fs, key=os.path.getmtime):
            if not f.endswith(".png"):
                continue
            try:
                int(os.path.basename(f).split(".")[0])
            except ValueError:
                continue
            imgs.append(Image.open(f))
        path = "{}/generating.gif".format(folder)
        if os.path.exists(path):
            os.remove(path)
        img = Image.new(imgs[0].mode, imgs[0].size, color=(255, 255, 255, 255))
        img.save(path, append_images=imgs, optimize=False, save_all=True, duration=400, loop=0)
        print("saved ", path)


if __name__ == "__main__":
    # save_gan("test", np.random.random((64, 28, 28, 1)), 0, np.arange(0, 64))
    # cgan_res()
    # save_infogan(None, 1)
    # infogan_comp()
    cvt_gif(["wgangp", "wgan", "infogan", "cgan", "acgan", "gan"])