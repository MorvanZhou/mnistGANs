# [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class G(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.mnist_uni_gen_cnn = mnist_uni_gen_cnn(input_shape)
    def forward(self, x):
        return self.mnist_uni_gen_cnn(x)
    
class D(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.mnist_uni_disc_cnn = mnist_uni_disc_cnn(input_shape)
        self.layer_out = nn.Linear(6272, 1)
    def forward(self, x):
        x = self.mnist_uni_disc_cnn(x)
        x = self.layer_out(x)
        x = torch.tanh(x)
        return x

class DCGAN(nn.Module):
    def __init__(self, latent_dim, img_shape, batch_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.batch_size = batch_size

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt_g = Adam(self.g.parameters(), lr=0.0001)
        self.opt_d = Adam(self.d.parameters(), lr=0.0001)

    def forward(self, batch_size, training=None):
        if training:
            self.g.train()
        else:
            self.g.eval()
        return self.g(torch.randn((batch_size, self.latent_dim),device=device))

    def _get_generator(self):
        model = G(input_shape=self.latent_dim)
        model = model.to(device)
        summary(model, input_size=(self.latent_dim,))
        return model

    def _get_discriminator(self):
        model = D(input_shape=self.img_shape)
        model = model.to(device)
        summary(model, input_size=self.img_shape)
        return model

    def train_d(self, data_real, data_gen):
        self.d.train()
        self.opt_d.zero_grad() 
        pred_real = self.d(data_real)
        pred_gen = self.d(data_gen)
        loss = F.relu(1.-pred_real).mean() + F.relu(1.+pred_gen).mean()
        loss.backward()
        self.opt_d.step()
        return loss

    def train_g(self, data_gen):
        self.g.train()
        self.opt_g.zero_grad() 
        pred_gen = self.d(data_gen)
        loss = F.relu(1.-pred_gen).mean()
        loss.backward()
        self.opt_g.step()
        return loss

    def step(self, data):
        data_real = torch.tensor(data).to(device)
        data_gen = self.forward(self.batch_size)
        d_loss = self.train_d(data_real, data_gen.detach())  #梯度的计算截止到data_gen，不再反向传回generator
     
        # data_gen = self.forward(self.batch_size)
        g_loss = self.train_g(data_gen)
        
        return d_loss,  g_loss
    


def train(gan, ds, epoch):
    t0 = time.time()
    for ep in range(epoch):
        for t, (img, labels) in enumerate(ds):
            d_loss, g_loss= gan.step(img.to(torch.float32))
            if t % 400 == 0:
                t1 = time.time()
                print(
                    "ep={} | time={:.1f} | t={} | d_loss={:.2f} | g_loss={:.2f}".format(
                        ep, t1 - t0, t, d_loss.cpu().detach().numpy(), g_loss.cpu().detach().numpy() ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    LATENT_DIM = 100
    IMG_SHAPE = (1, 28, 28)
    BATCH_SIZE = 64
    EPOCH = 20
    
    train_loader, test_loader = load_mnist(BATCH_SIZE)
    m = DCGAN(LATENT_DIM, IMG_SHAPE, BATCH_SIZE)
    train(m, train_loader, EPOCH)



