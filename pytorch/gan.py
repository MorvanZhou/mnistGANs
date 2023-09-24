# [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from visual import save_gan, cvt_gif
from utils import save_weights
import numpy as np
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_real_data(data_dim, batch_size):
    for i in range(300):
        a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis].astype(np.float32)
        base = np.linspace(-1, 1, data_dim)[np.newaxis, :].repeat(batch_size, axis=0).astype(np.float32)
        yield a * np.power(base, 2) + (a-1)



class GAN(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.g = self._get_generator().to(device)
        self.d = self._get_discriminator().to(device)

        self.opt_g = Adam(self.g.parameters(), lr=0.0001)
        self.opt_d = Adam(self.d.parameters(), lr=0.0001)

    def forward(self, n, training=True):
        if training:
            self.g.train()
        else:
            self.g.eval()
        return self.g(torch.randn((n,self.latent_dim),device=device))

    def _get_generator(self):
        model = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.data_dim),
        )
        print(model)
        return model

    def _get_discriminator(self):
        model = nn.Sequential(
            nn.Linear(self.data_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        print(model)
        return model
    
    # 相较于莫烦的打标签，我认为用hinge gan loss更便于理解
    def train_d(self, data_real, data_gen):
        self.opt_d.zero_grad() 
        pred_real = self.d(data_real)
        pred_gen = self.d(data_gen)
        loss = torch.mean(F.relu(1. - pred_real) + F.relu(1. + pred_gen))
        loss.backward()
        self.opt_d.step()
        return loss

    def train_g(self, data_gen):
        self.opt_g.zero_grad() 
        pred_gen = self.d(data_gen)
        loss = torch.mean(F.relu(1. - pred_gen))
        loss.backward()
        self.opt_g.step()
        return loss

    def step(self, data):
        data_real = torch.tensor(data).to(device)
        data_gen = self.forward(len(data_real))
        g_loss = self.train_g(data_gen)
        
        data_gen = self.forward(len(data_real))
        d_loss = self.train_d(data_real, data_gen.detach())  #梯度的计算截止到data_gen，不再反向传回generator
        return d_loss,  g_loss


def train(gan, epoch):
    t0 = time.time()
    for ep in range(epoch):
        for t, data in enumerate(get_real_data(DATA_DIM, BATCH_SIZE)):
            d_loss,  g_loss = gan.step(data)
            if t % 400 == 0:
                t1 = time.time()
                print(
                    "ep={} | time={:.1f} | t={} | d_loss={:.2f} | g_loss={:.2f}".format(
                        ep, t1 - t0, t, d_loss.detach().cpu().numpy(), g_loss.detach().cpu().numpy(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan, shrink=2)


if __name__ == "__main__":
    LATENT_DIM = 10
    DATA_DIM = 16
    BATCH_SIZE = 32
    EPOCH = 40

    m = GAN(LATENT_DIM, DATA_DIM)
    train(m, EPOCH)


