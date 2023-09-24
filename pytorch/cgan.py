# [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torchsummary import summary
from utils import save_weights
from visual import save_gan, cvt_gif
from mnist_ds import load_mnist
from gan_cnn import mnist_uni_disc_cnn, mnist_uni_gen_cnn
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class D(nn.Module):
    def __init__(self, img_shape):
        super(D, self).__init__()

        # Define the layers for label embedding
        self.emb_layer = nn.Sequential(
            nn.Embedding(10, 32),
            nn.Linear(32, 28*28),
            nn.ReLU()
        )

        # Define the layers for image processing
        self.img_processing = mnist_uni_disc_cnn(input_shape=(2,28,28))

        # Output layer
        self.output_layer = nn.Linear(6272, 1)

    def forward(self, img, label):
        # Label embedding
        label_emb = self.emb_layer(label)
        label_emb = label_emb.reshape(label_emb.shape[0], 1,28,28)

        # Concatenate image and label embedding
        concat_img = torch.cat((img, label_emb), dim=1)
        img_output = self.img_processing(concat_img)

        # Pass through the output layer
        y = self.output_layer(img_output)
        # y = torch.tanh(y)  tanh不是必要的，实际上在很多成熟模型中都没有tanh，当然加上也没事
        return y

class G(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super(G, self).__init__()

        # Calculate the input dimension for the generator model
        input_dim = latent_dim + label_dim

        # Define the generator layers
        self.generator = mnist_uni_gen_cnn(input_shape=input_dim)

    def forward(self, noise, label):
        # Concatenate noise and one-hot encoded label
        label_onehot = F.one_hot(label, num_classes=10)
        model_in = torch.cat((noise, label_onehot), dim=1)

        # Pass through the generator model
        output = self.generator(model_in)

        return output

class CGAN(nn.Module):
    def __init__(self, latent_dim, label_dim, img_shape, batch_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.img_shape = img_shape
        self.batch_size = batch_size

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt_g = Adam(self.g.parameters(), lr=0.0001)
        self.opt_d = Adam(self.d.parameters(), lr=0.0001)

    def _get_generator(self):
        model = G(self.latent_dim, self.label_dim)
        model = model.to(device)
        return model

    def _get_discriminator(self):
        model = D(self.img_shape)
        model = model.to(device)
        return model

    def forward(self, target_labels):
        noise = torch.randn(len(target_labels), self.latent_dim, device=device)
        target_labels = target_labels.to(torch.int64)
        return self.g.forward(noise, target_labels)

    def train_g(self, random_img_label):
        self.g.train()
        self.opt_g.zero_grad()
        img_fake = self.forward(random_img_label)
        pred_fake = self.d(img_fake, random_img_label)
        loss = F.relu(1.-pred_fake).mean()
        loss.backward()
        self.opt_g.step()
        return loss, img_fake

    def train_d(self, img_real, img_fake, img_real_label, img_fake_label):
        self.d.train()
        self.opt_d.zero_grad()
        pred_real = self.d(img_real, img_real_label)
        pred_fake = self.d(img_fake, img_fake_label)
        loss = F.relu(1.-pred_real).mean() + F.relu(1.+pred_fake).mean()
        loss.backward()
        self.opt_d.step()
        return loss

    def step(self, img_real, img_real_label):
        img_fake_label = torch.randint(0, 10, (len(img_real),), dtype=torch.int32, device=device)
        g_loss, img_fake = self.train_g(img_fake_label)
        d_loss = self.train_d(img_real, img_fake.detach(), img_real_label, img_fake_label)

        return img_fake, d_loss, g_loss


def train(gan, ds, EPOCH):
    t0 = time.time()
    for ep in range(EPOCH):
        for t, (real_img, real_img_label) in enumerate(ds):
            real_img, real_img_label = real_img.to(device), real_img_label.to(device)
            g_img, d_loss, g_loss = gan.step(real_img, real_img_label)
            if t % 400 == 0:
                t1 = time.time()
                print("ep={} | time={:.1f} | t={} | d_loss={:.2f} | g_loss={:.2f}".format(
                    ep, t1-t0, t, d_loss.item(), g_loss.item(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan)

if __name__ == "__main__":
    LATENT_DIM = 100
    IMG_SHAPE = (1, 28, 28)
    LABEL_DIM = 10
    BATCH_SIZE = 64
    EPOCH = 60

    # set_soft_gpu(True)
    # d = get_half_batch_ds(BATCH_SIZE)
    # m = CGAN(LATENT_DIM, LABEL_DIM, IMG_SHAPE)
    # train(m, d)
    train_loader, test_loader = load_mnist(BATCH_SIZE)
    m = CGAN(LATENT_DIM, LABEL_DIM, IMG_SHAPE, BATCH_SIZE)
    train(m, train_loader, EPOCH)






