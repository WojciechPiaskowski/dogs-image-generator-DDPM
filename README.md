# Dogs Image Generator (DDPM) - Diffusion Model

# Overview

The project's goal is to create a Denoising Diffusion Probabilistic Model (DDPM) from scratch (not utilizing DDPM implementations from libraries, only pytorch) to generate images of dogs.

DDPM is an image generator model that achieves similar or better results than different implementations of Generative adversarial networks (GANs) and Variational Autoencoders.
The idea is to feed the model with images with some amount of noise (different amounts at each sample) and train the model to predict that noise or at least some of it.
This eventually leads to a model that is decent at predicting noise pixels within an image.

With a trained model, pure noise can be fed as input to predict some amount of that noise, subtract it, and then feed it into the model again. Performing this process many times (in this case 1'000), the model generates a brand new image, that should be similar to the distribution the model has learned - dogs in this case.

# How to run

Install required python libraries (refer to requirements.txt).

Run image scrapper to download dog images from google image search. I've downloaded in total around 14'000 images across 26 dog breeds using Selenium.

``` python image_scrapper.py ```

Once images are downloaded and properly stored (data/dogs/breed/) run the model training.

``` python train.py ```

You can specify the following arguments:
```
--batch_size (default: 16) -> training batch size
--examples (default: False) -> show an example of dog images from the data loader and an example of applying forward diffusion to an image
--T (default: 1000) -> number of diffusion steps (both for training and inference)
--device (default: 'cuda') -> pytoch training resource. 'cuda' will use a GPU, change to 'cpu' if needed
--lr (default: 0.0003) -> training learning rate
--epochs (default: 500) -> number of epochs to run
--samples (default: 10) -> number of samples generated and saved every save_frequency
--save_frequency (default: 10) -> how many epochs between generating samples and saving model states (training can be stopped and resumed at a later time)
```

Training with nondefault parameters.
``` python train.py --batch_size 32 --save_frequency 5 --T 2000``` 

# Noise scheduler

Both linear and cosine beta (noise) schedulers are implemented. It would seem that the cosine scheduler should yield better results as it is applying the noise slower (there are more timesteps when the image is very visible).
In the end, a model trained on images that were processed by a linear beta scheduler look more like dogs, though this is subjective, they do look different.
Sample results from models trained on both are displayed in the **results** section.

# Model Architecture

# UNet
The model uses the **UNet** architecture that consists of **downsample**, **upsample** and **self-attention blocks**.
The model's inputs are **processed dog images** and **timestep embeddings**.

**Image processing** includes:
- resizing them to 64x64 pixels
- random horizontal flip
- scaling the pixel values to [-1, 1]
- loading images from folders to a pytorch DataLoader

**Timestep embedding** uses sinus and cosine functions to map integer timestep values (0 to T, in this case, 0 to 1000) to a continuous space with specified embedding size (in this case 256).
This timestep embedding is later passed within downsample and umpsample layers.
  

# Downsampling block
Downsampling block uses max pooling to decrease the image size, and encode the information, while simultaneously the convolution layers following the max pooling increase the number of feature maps that are passed on further into the network.
In each downsample layer there are in total 4 layers of 2d convolution that are mixed up with 2 GELU activation functions and 4 group normalization layers.

Timestep embedding is processed separately, instead of going through the convolution layers it passes through a SILU activation function and into a linear fully connected layer.
The Ooutput of this linear network (time embedding) is added to convolution layers output (feature maps).

# Upsampling block
Similar to downsampling blocks, however, they upsampling layers instead of max pooling to increase the image size, while also decreasing the number of feature maps. They work as a decoder to the downsampling encoder of information. They also contain a total of 4 2d convolution layers mixed with normalization and activation functions.
Additionally the upsampling blocks use **residual connections**, which prove to be useful in in deeper networks, and specifically useful for reconstructing images.

Otherwise, upsampling blocks use the same convolution layers, normalization and linear embedding processing that is later added to the convolution output.

# Self-attention blocks
Self-attention layers are typically useful for building context between inputs. In this case, they were added experimentally, and in the end improved the output significantly.

Query, Key and Value are all normalized copies of the input to the 4-head attention layer. That is later passed through 2 linear layers mixed with layer normalization and GELU activation function.

# Final structure
The network consists of:
- 2 convolution layers
- 4 downsample blocks
- 6 middle convolution layers
- 4 upsample blocks
- output layer - output shape is batch_size x 64 x 64 x 3

The model consists of 93 417 859 parameters.

# Loss function
The loss function used is the mean squared error (MSE) between the noise and the predicted noise

# Exponential moving average (EMA)
The original model is also copied and updated as 0.95 * previous weights + 0.05 * new weights to smooth out the learning process and reduce the impact of outliers.
This however did not bring improvements to the model output.

# Results

**samples generated from linear beta scheduler trained model**
![v4_1](https://github.com/WojciechPiaskowski/dogs-image-generator-DDPM/assets/57685224/c45fff77-f5f5-4bd2-8655-a5aaef81d860)
![v4_2](https://github.com/WojciechPiaskowski/dogs-image-generator-DDPM/assets/57685224/d5e69da5-86ec-4977-b8d6-afc3b62e03ef)

**samples generated from cosine beta scheduler trained model**
![ema_epoch_370v5_ema](https://github.com/WojciechPiaskowski/dogs-image-generator-DDPM/assets/57685224/dc057238-8bd8-4dcd-8e86-962007d0e478)
![epoch_470v5](https://github.com/WojciechPiaskowski/dogs-image-generator-DDPM/assets/57685224/ff22a0eb-9012-4124-bce7-51dbd7014454)

# Possible improvements

- delete unnecessary images -> many images in the scrapped dataset contain a collage of dogs, which results in the model generating a lot of very small dogs in an image, which in many cases is hard to even see in 64x64 resolution
- Classifier guidance -> add classes so the model learns to generate specific types of dogs
- larger model -> larger model, more compute
- more images -> larger dataset could improve the results
