import cv2
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

def load_transformed_dataset(path, img_size=64, batch_size=128):

    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: (x * 2) - 1)  # scales data to [-1, 1]
    ]

    data_transform = transforms.Compose(data_transforms)

    ds = ImageFolder(root=path, transform=data_transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dl


def show_images(dl, num_samples=20, cols=4):

    for data, y in dl:
        break

    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        random_idx = np.random.randint(0, data.shape[0])
        img = (data[random_idx] + 1) / 2
        plt.subplot(int(num_samples / cols + 1), cols, i + 1)
        plt.imshow(img.permute(1, 2, 0))

    return

dl = load_transformed_dataset(path='data/cars')
show_images(dl)


#forward process - adding noise / noise scheduler


def linear_beta_scheduler(timesteps, start=0.0001, end=0.02):

    out = torch.linspace(start, end, timesteps)

    return out


# returns an index t of list given the batch dimension
# TODO: undesrtand what this is used for
def get_index_from_list(vals, t, x_shape):

    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    return out

# TODO: understand what this is used for
# takes an image and a timestep, retuns noisy image at that timestep
def forward_diffusion_sample(sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, x0, t, device='cpu'):

    noise = torch.randn_like(x0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x0.shape)

    # mean + variance
    noisy_img = sqrt_alphas_cumprod_t.to(device) * x0.to(device) \
          + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    noise = noise.to(device)

    return noisy_img, noise


# beta scheduler
T = 120
betas = linear_beta_scheduler(timesteps=T)

# pre-calculate terms, alphas cumulative produts
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


# simulate forward diffusion over single image
img = next(iter(dl))[0][0]
img = (img + 1) / 2

plt.figure(figsize=(15, 15))
plt.axis('off')
n_images = 10
step_size = int(T / n_images)

for idx in range(0, T, step_size):

    t = torch.tensor([idx]).type(torch.int64)
    plt.subplot(1, n_images+1, int((idx/step_size)+1))
    img, noise = forward_diffusion_sample(sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, img, t)
    plt.imshow(img.permute(1, 2, 0))

plt.show()

# U-NET - used for backward diffusion
# convlolutions, down and up sampling, residual connections
# similar to auto-encoder
# denoising score matching
# positional embeddings are used for the step in the sequence information (t)
# U-Net needs to predict the noise and subtract it from the image (to get image at noise step t-1)

















