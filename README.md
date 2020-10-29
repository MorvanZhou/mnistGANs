# GANs implementation using MNIST data

This repo is a collection of the implementations of many GANs. 
In order to make the codes easy to read and follow,
 I minimize the code and run on the same MNIST dataset.

What does the MNIST data look like?

![](https://mofanpy.com/static/results/gan/mnist.png)

Toy implementations are organized as following:

**1. Base Method**

- [Deep Convolutional GAN (DCGAN)](#DCGAN) 

**2. Loss or Structure Modifications**

- [Least Squares GAN (LSGAN)](#LSGAN)
- [Wasserstein GAN (WGAN)](#WGAN) 
    - [Gradient Penalty (WGAN gp)](#WGANpg) 
    - [Wasserstein Divergence (WGAN_div)](#WGANdiv)
- [Self-Attention GAN](#SAGAN)
- [Progressive-Growing GAN](#PGGAN)
    
**3. Can be Conditional**

- [Conditional GAN (CGAN)](#CGAN) 
- [Auxiliary Classifier GAN (ACGAN)](#ACGAN) 
- [InfoGAN](#InfoGAN) 

**4. Image to Image Transformation**

- [Context-Conditional GAN (CCGAN)](#CCGAN)
- [CycleGAN](#CycleGAN)
- [Pix2Pix](#Pix2Pix)
- [Super-Resolution GAN](#SRGAN)
- [StyleGAN](#StyleGAN)
- WIP
 
# Installation
```shell script
$ git clone https://github.com/MorvanZhou/mnistGANs
$ cd mnistGANs/
$ pip3 install -r requirements.txt
```

 
## DCGAN
[Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)

[code](gan.py)

![](https://mofanpy.com/static/results/gan/gan/generating.gif)
 
## LSGAN
[Least Squares Generative Adversarial Networks](https://arxiv.org/pdf/1611.04076.pdf)

[code](lsgan.py)

![](https://mofanpy.com/static/results/gan/lsgan/generating.gif)

## WGAN
[Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)

[code](wgan.py)

![](https://mofanpy.com/static/results/gan/wgan/generating.gif)

## WGANpg
[Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf)

[code](wgan_gp.py)

![](https://mofanpy.com/static/results/gan/wgangp/generating.gif)

## WGANdiv
[Wasserstein Divergence for GANs](https://arxiv.org/pdf/1712.01026.pdf)

[code](wgan_div.py)

![](https://mofanpy.com/static/results/gan/wgandiv/generating.gif)

## SAGAN
[Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318.pdf)

[code](sagan.py)

![](https://mofanpy.com/static/results/gan/sagan/generating.gif)

## PGGAN
[PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](https://arxiv.org/pdf/1710.10196.pdf)

[code](pggan.py)

![](https://mofanpy.com/static/results/gan/pggan/generating.gif)

## CGAN
[Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

[code](cgan.py)

![](https://mofanpy.com/static/results/gan/cgan/generating.gif)

## ACGAN
[Conditional Image Synthesis with Auxiliary Classifier GANs](https://arxiv.org/pdf/1610.09585.pdf)

[code](acgan.py)

![](https://mofanpy.com/static/results/gan/acgan/generating.gif)

## InfoGAN
[InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/pdf/1606.03657.pdf)

[code](infogan.py)

![](https://mofanpy.com/static/results/gan/infogan/generating.gif)


## CCGAN
[Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1611.06430.pdf)

[code](ccgan.py)

![](https://mofanpy.com/static/results/gan/ccgan/generating.gif)

## CycleGAN
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593)

[code](cyclegan.py)

![](https://mofanpy.com/static/results/gan/cyclegan/generating.gif)

## Pix2Pix
[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)

[code](pix2pix.py)

![](https://mofanpy.com/static/results/gan/pix2pix/generating.gif)

## SRGAN
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)

[code](srgan.py)

![](https://mofanpy.com/static/results/gan/srgan/generating.gif)

## StyleGAN
[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)

[code](stylegan.py)

![](https://mofanpy.com/static/results/gan/stylegan/generating.gif)