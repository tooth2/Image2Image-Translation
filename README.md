# Image2Image-Translation
A PyTorch implementation using CycleGAN architecture, to read in an image from a set  X  and transform it so that it looks as if it belongs in set  Y .

## The goal
Input data set is a set of images of Yosemite national park taken either during the summer of winter. The objective will be to train generators that learn to transform an image from domain  X  into an image that looks like it came from domain  Y  (and vice versa).
![objective](images/XY_season_images.png)

## CycleGAN
A CycleGAN is designed for image-to-image translation and it learns from unpaired training data. This means that in order to train a generator to translate images from domain X to domain Y, we do not have to have exact correspondences between individual images in those domains. For example, in the [paper that introduced CycleGANs](https://arxiv.org/pdf/1703.10593.pdf), the authors are able to translate between images of horses and zebras, even though there are no images of a zebra in exactly the same position as a horse or with exactly the same background, etc. Thus, CycleGANs enable learning a mapping from one domain  X  to another domain  Y  without having to find perfectly-matched, training pairs.

![Horse2Zebra](horse2zebra.jpg)

A CycleGAN is made of two types of networks: discriminators, and generators. In this example, the discriminators are responsible for classifying images as real or fake (for both  X  and  Y  kinds of images). The generators are responsible for generating convincing, fake images for both kinds of images.

![Cycle GAN](images/cycle_consistency_ex.png)

## Implementation Step
1. Load in the image data using PyTorch's DataLoader class to efficiently read in images from a specified directory.
2. Define the CycleGAN architecture; define the discriminator and the generator models.
3. Train by calculating the adversarial and cycle consistency losses for the generator and discriminator network and completing a number of training epochs. 
4. Evaluate the model by looking at the loss over time and looking at sample, generated images.

## Define the Model 
A CycleGAN is made of two discriminator and two generator networks.

### Discriminators
The discriminators,  DX  and  DY , in this CycleGAN are convolutional neural networks that see an image and attempt to classify it as real or fake. In this case, real is indicated by an output close to 1 and fake as close to 0. The discriminators have the following architecture:

![Discriminator](images/discriminator_layers.png)

This network sees a 128x128x3 image, and passes it through 5 convolutional layers that downsample the image by a factor of 2. The first four convolutional layers have a BatchNorm and ReLu activation function applied to their output, and the last acts as a classification layer that outputs one value.

### Generators
The generators, G_XtoY and G_YtoX (sometimes called F), are made of an encoder, a conv net that is responsible for turning an image into a smaller feature representation, and a decoder, a transpose_conv net that is responsible for turning that representation into an transformed image. These generators, one from XtoY and one from YtoX, have the following architecture:

![Generator](images/cyclegan_generator_ex.png)

This network sees a 128x128x3 image, compresses it into a feature representation as it goes through three convolutional layers and reaches a series of residual blocks. It goes through a few (typically 6 or more) of these residual blocks, then it goes through three transpose convolutional layers (sometimes called de-conv layers) which upsample the output of the resnet blocks and create a new image!

Note that most of the convolutional and transpose-convolutional layers have BatchNorm and ReLu functions applied to their outputs with the exception of the final transpose convolutional layer, which has a tanh activation function applied to the output. Also, the residual blocks are made of convolutional and batch normalization layers.

### The Resnet and Residual Blocks
To define the generators, a Residual Blocks which connect the encoder and decoder portions of the generators is introduced. Refering to  ResNet50 for image classification, as belows: 
![resnet](images/resnet_50.png)

ResNet blocks rely on connecting the output of one layer with the input of an earlier layer. The motivation for this structure is as follows: very deep neural networks can be difficult to train. Deeper networks are more likely to have vanishing or exploding gradients and, therefore, have trouble reaching convergence; batch normalization helps with this a bit. However, during training, we often see that deep networks respond with a kind of training degradation. Essentially, the training accuracy stops improving and gets saturated at some point during training. In the worst cases, deep models would see their training accuracy actually worsen over time!

One solution to this problem is to use Resnet blocks that allow us to learn so-called residual functions as they are applied to layer inputs. In the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) this residual learning: a building block is introduced. 

![Residual Blocks](images/resnet_block.png)

### The Discriminator and The Generator Losses 
Computing the discriminator and the generator losses are key to getting a CycleGAN to train.
![loss](images/CycleGAN_loss.png)

* The CycleGAN contains two mapping functions  G:X→Y  and  F:Y→X , and associated adversarial discriminators  DY  and  DX . 
    * (a)  DY  encourages  G  to translate  X  into outputs indistinguishable from domain  Y , and vice versa for  DX  and  F .
* The two cycle consistency losses that capture the intuition that when translating from one domain to the other and back again  
    * (b) Forward cycle-consistency loss 
    * (c) backward cycle-consistency loss.
* The Loss functions
    * `real_mse_loss` (mean sqaured error) that looks at the output of a discriminator and returns the error based on how close that output is to being classified as real.
    * `fake_mse_loss` (mean sqaured error) that looks at the output of a discriminator and returns the error based on how close that output is to being classified as fake.     * `cycle_consistency_loss` that looks at a set of real image and a set of reconstructed/generated images, and returns the mean absolute error between them. 
    * `lambda_weight` a parameter that will weight the mean absolute error in a batch. Refer the original paper about a starting value for this lambda_weight.

## Training a CycleGAN
When a CycleGAN trains, and sees one batch of real images from set  X  and  Y , it trains by performing the following steps:

![training](images/training_cycle_ex.png)

### Training the Discriminators
1. Compute the discriminator  DX  loss on real images
2. Generate fake images that look like domain  X  based on real images in domain  Y 
3. Compute the fake loss for  DX 
4. Compute the total loss and perform backpropagation and  DX  optimization
5. Repeat steps 1-4 only with  DY  and your domains switched!

### Training the Generators
1. Generate fake images that look like domain  X  based on real images in domain  Y 
2 Compute the generator loss based on how  DX  responds to fake  X 
3. Generate reconstructed  Ŷ   images based on the fake  X  images generated in step 1
4. Compute the cycle consistency loss by comparing the reconstructions with real  Y  images
5. Repeat steps 1-4 only swapping domains
6. Add up all the generator and reconstruction losses and perform backpropagation + optimization

## Reference 
* [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
* [Image Synthesis using GAN](https://arxiv.org/pdf/1711.11585.pdf)
* [Unpaired Image to Image Translation](https://arxiv.org/pdf/1703.10593.pdf)
* [Least Squares Generative Adversarial Networks](https://arxiv.org/pdf/1611.04076.pdf)
* [PyTorch](https://pytorch.org/docs/stable/optim.html#algorithms)
* [Cycle GAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)
* [NVIDIA pix2pix](https://github.com/NVIDIA/pix2pixHD)
* [Unsupervised image translation](https://github.com/mingyuliutw/UNIT)
