# imports
import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


# create dataloader object from images in folders, transform them
def load_transformed_dataset(path, img_size=64, batch_size=16):

    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: (x * 2) - 1)  # scales data to [-1, 1]
    ]

    data_transform = transforms.Compose(data_transforms)

    ds = ImageFolder(root=path, transform=data_transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return dl


# function to display images from the data loader
def show_images(dl, num_samples=20, cols=4, display=False, save=False, title=''):

    if not display:
        plt.ioff()

    plt.figure(figsize=(15, 15))
    for i in range(num_samples):

        # get a random batch from data loader
        for x, y in dl:
            break

        random_idx = np.random.randint(0, x.shape[0])
        img = (x[random_idx] + 1) / 2
        plt.subplot(int(num_samples / cols + 1), cols, i + 1)
        plt.imshow(img.permute(1, 2, 0))

    if save:
        plt.savefig(f'samples/{title}.png')

    plt.ion()

    return


# retrieves the value at specific timestep from pytorch tensor
def get_index_from_list(vals, t, x_shape):

    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to('cuda')
    return out


# forward process - adding noise / noise scheduler
def cosine_beta_schedule(timesteps, s=0.008):

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):

    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)
