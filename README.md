# GANs implementations using MNIST data


**1. No condition**

- [Deep Convolutional GAN (DCGAN)](#DCGAN) 
- [Least Squares GAN (LSGAN)](#LSGAN)
- [Wasserstein GAN (WGAN)](#WGAN) 
    - [Gradient Penalty (WGAN gp)](#WGANpg) 
    - [Wasserstein Divergence (WGAN_div)](#WGANdiv)
    
**2. With some condition**

- [Conditional GAN (CGAN)](#CGAN) 
- [Auxiliary Classifier GAN (ACGAN)](#ACGAN) 
- [InfoGAN](#InfoGAN) 

**3. Change information in picture**

- [Context-Conditional GAN (CCGAN)](#CCGAN)
- [CycleGAN](#CycleGAN)
- WIP
 
 
## DCGAN
[Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[code](gan.py)

![](visual/gan/generating.gif)
 
## LSGAN
[Least Squares Generative Adversarial Networks](https://arxiv.org/pdf/1611.04076.pdf)

[code](lsgan.py)

![](visual/lsgan/generating.gif)

## WGAN
[Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)

[code](wgan.py)

![](visual/wgan/generating.gif)

## WGANpg
[Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf)

[code](wgan_gp.py)

![](visual/wgangp/generating.gif)

## WGANdiv
[Wasserstein Divergence for GANs](https://arxiv.org/pdf/1712.01026.pdf)

[code](wgan_div.py)

![](visual/wgandiv/generating.gif)

## CGAN
[Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

[code](cgan.py)

![](visual/cgan/generating.gif)

## ACGAN
[Conditional Image Synthesis with Auxiliary Classifier GANs](https://arxiv.org/pdf/1610.09585.pdf)

[code](acgan.py)

![](visual/acgan/generating.gif)

## InfoGAN
[InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/pdf/1606.03657.pdf)

[code](infogan.py)

![](visual/infogan/generating.gif)

## CCGAN
[Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1611.06430.pdf)

[code](ccgan.py)

![](visual/ccgan/generating.gif)

## CycleGAN
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593)

[code](cyclegan.py)

![](visual/cyclegan/generating.gif)