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
    if x.ndim > 3:
        x = np.squeeze(x, axis=-1)
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


def save_gan(model, ep, **kwargs):
    name = model.__class__.__name__.lower()
    if name in ["dcgan", "wgan", "wgangp", "lsgan", "wgandiv", "sagan", "pggan"]:
        imgs = model.call(100, training=False).numpy()
        _save_gan(name, ep, imgs, show_label=False)
    elif name == "gan":
        data = model.call(5, training=False).numpy()
        plt.plot(data.T)
        plt.xticks((), ())
        dir_ = "visual/{}".format(name)
        os.makedirs(dir_, exist_ok=True)
        path = dir_ + "/{}.png".format(ep)
        plt.savefig(path)
    elif name == "cgan" or name == "acgan":
        img_label = np.arange(0, 10).astype(np.int32).repeat(10, axis=0)
        imgs = model.predict(img_label)
        _save_gan(name, ep, imgs, show_label=True)
    elif name in ["infogan"]:
        img_label = np.arange(0, model.label_dim).astype(np.int32).repeat(10, axis=0)
        img_style = np.concatenate(
            [np.linspace(-model.style_scale, model.style_scale, 10)] * 10).reshape((100, 1)).repeat(model.style_dim, axis=1).astype(np.float32)
        img_info = img_label, img_style
        imgs = model.predict(img_info)
        _save_gan(name, ep, imgs, show_label=False)
    elif name in ["ccgan", "pix2pix"]:
        if "img" not in kwargs:
            raise ValueError
        input_img = kwargs["img"][:100]
        mask_width = np.random.randint(model.mask_range[0], model.mask_range[1], len(input_img))
        mask = np.ones(input_img.shape, np.float32)
        for i, w in enumerate(mask_width):
            mask_xy = np.random.randint(0, model.img_shape[0] - w, 2)
            x0, x1 = mask_xy[0], w + mask_xy[0]
            y0, y1 = mask_xy[1], w + mask_xy[1]
            mask[i, x0:x1, y0:y1] = 0
        masked_img = input_img * mask
        imgs = model.predict(masked_img)
        _save_img2img_gan(name, ep, masked_img, imgs)
    elif name == "cyclegan":
        if "img6" not in kwargs or "img9" not in kwargs:
            raise ValueError
        img6, img9 = kwargs["img6"][:50], kwargs["img9"][:50]
        img9_, img6_ = model.g12.call(img6, training=False), model.g21.call(img9, training=False)
        img = np.concatenate((img6.numpy(), img9.numpy()), axis=0)
        imgs = np.concatenate((img9_.numpy(), img6_.numpy()), axis=0)
        _save_img2img_gan(name, ep, img, imgs)
    elif name in ["srgan"]:
        if "img" not in kwargs:
            raise ValueError
        input_img = kwargs["img"][:100]
        imgs = model.predict(input_img)
        _save_img2img_gan(name, ep, input_img, imgs)
    elif name == "stylegan":
        n = 12
        global z1, z2       # z1 row, z2 col
        if "z1" not in globals():
            z1 = np.random.normal(0, 1, size=(n, 1, model.latent_dim))
        if "z2" not in globals():
            z2 = np.random.normal(0, 1, size=(n, 1, model.latent_dim))
        imgs = model.predict([
            np.concatenate(
                (z1.repeat(n, axis=0).repeat(1, axis=1), np.repeat(np.concatenate([z2 for _ in range(n)], axis=0), 2, axis=1)),
                axis=1),
            np.zeros([len(z1)*n, model.img_shape[0], model.img_shape[1]], dtype=np.float32)])
        z1_imgs = -model.predict([z1.repeat(model.n_style, axis=1), np.zeros([len(z1), model.img_shape[0], model.img_shape[1]], dtype=np.float32)])
        z2_imgs = -model.predict([z2.repeat(model.n_style, axis=1), np.zeros([len(z2), model.img_shape[0], model.img_shape[1]], dtype=np.float32)])
        imgs = np.concatenate([z2_imgs, imgs], axis=0)
        rest_imgs = np.concatenate([np.ones([1, 28, 28, 1], dtype=np.float32), z1_imgs], axis=0)
        for i in range(len(rest_imgs)):
            imgs = np.concatenate([imgs[:i*(n+1)], rest_imgs[i:i+1], imgs[i*(n+1):]], axis=0)
        _save_gan(name, ep, imgs, show_label=False, nc=n+1, nr=n+1)
    else:
        raise ValueError(name)
    plt.clf()
    plt.close()

def _img_recenter(img):
    return (img + 1) * 255 / 2


def _save_img2img_gan(model_name, ep, img1, img2):
    if not isinstance(img1, np.ndarray):
        img1 = img1.numpy()
    if not isinstance(img2, np.ndarray):
        img2 = img2.numpy()
    if img1.ndim > 3:
        img1 = np.squeeze(img1, axis=-1)
    if img2.ndim > 3:
        img2 = np.squeeze(img2, axis=-1)
    img1, img2 = _img_recenter(img1), _img_recenter(img2)
    plt.clf()
    nc, nr = 20, 10
    plt.figure(0, (nc * 2, nr * 2))
    i = 0
    for c in range(0, nc, 2):
        for r in range(nr):
            n = r * nc + c
            plt.subplot(nr, nc, n + 1)
            plt.imshow(img1[i], cmap="gray")
            plt.axis("off")
            plt.subplot(nr, nc, n + 2)
            plt.imshow(img2[i], cmap="gray_r")
            plt.axis("off")
            i += 1

    plt.tight_layout()
    dir_ = "visual/{}".format(model_name)
    os.makedirs(dir_, exist_ok=True)
    path = dir_ + "/{}.png".format(ep)
    plt.savefig(path)


def _save_gan(model_name, ep, imgs, show_label=False, nc=10, nr=10):
    if not isinstance(imgs, np.ndarray):
        imgs = imgs.numpy()
    if imgs.ndim > 3:
        imgs = np.squeeze(imgs, axis=-1)
    imgs = _img_recenter(imgs)
    plt.clf()
    plt.figure(0, (nc * 2, nr * 2))
    for c in range(nc):
        for r in range(nr):
            i = r * nc + c
            plt.subplot(nr, nc, i + 1)
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
        if imgs.ndim > 3:
            imgs = np.squeeze(imgs, axis=-1)
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


def cvt_gif(folders_or_gan, shrink=10):
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
            img = Image.open(f)
            img = img.resize((img.width//shrink, img.height//shrink), Image.ANTIALIAS)
            imgs.append(img)
        path = "{}/generating.gif".format(folder)
        if os.path.exists(path):
            os.remove(path)
        imgs[-1].save(path, append_images=imgs, optimize=False, save_all=True, duration=400, loop=0)
        print("saved ", path)


if __name__ == "__main__":
    # show_mnist(20)
    # cgan_res()
    # save_infogan(None, 1)
    # infogan_comp()
    cvt_gif(["wgangp", "wgandiv", "wgan", "cgan", "acgan", "dcgan", "lsgan", "infogan", "ccgan", "cyclegan", "pix2pix", "stylegan"])