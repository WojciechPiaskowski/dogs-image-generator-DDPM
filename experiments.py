import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.transforms import transforms
from torchvision.transforms.functional import adjust_brightness
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image
import os


img_size = 64
batch_size = 16

# remove corrupted image files
rt_path = f'{os.getcwd()}\\data\\dogs'
for subdir, dirs, files in os.walk(rt_path):
    for file in files:
        path = os.path.join(subdir, file)

        try:
            img = Image.open(path)
        except:
            os.remove(path)


def load_transformed_dataset(path, img_size=img_size, batch_size=batch_size):
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


def show_images(dl, num_samples=20, cols=4, display=False, save=False, title=''):

    if not display:
        plt.ioff()

    for data, y in dl:
        break

    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        random_idx = np.random.randint(0, data.shape[0])
        img = (data[random_idx] + 1) / 2
        plt.subplot(int(num_samples / cols + 1), cols, i + 1)
        plt.imshow(img.permute(1, 2, 0))

    if save:
        plt.savefig(f'samples/{title}.png')

    plt.ion()

    return

dl = load_transformed_dataset(path='data/dogs')
# show_images(dl, display=True, save=False, title='sample_images')

# forward process - adding noise / noise scheduler
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
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


# returns an index t of list given the batch dimension
# TODO: undesrtand what this is used for
def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to('cuda')

    return out


# takes an image and a timestep, retuns noisy image (and noise itself) at that timestep
def forward_diffusion_sample(x0, t, device='cuda'):
    noise = torch.randn_like(x0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x0.shape)


    # mean + variance
    noisy_img = sqrt_alphas_cumprod_t.to(device) * x0.to(device) \
                + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    noise = noise.to(device)

    return noisy_img, noise


# beta scheduler
T = 1000
betas = linear_beta_schedule(timesteps=T)
# betas = cosine_beta_schedule(timesteps=T)

# pre-calculate terms, alphas cumulative produts
alphas = 1.0 - betas
alphas_prev = F.pad(alphas[:-1], (1, 0), value=1.0)
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

# simulate forward diffusion over single image
def display_forward_diffusion(device='cuda'):

    img = next(iter(dl))[0][0]
    img = (img + 1) / 2

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    n_images = 10
    step_size = int(T / n_images)

    for idx in range(0, T, step_size):

        t = torch.tensor([idx]).type(torch.int64)
        plt.subplot(1, n_images+1, int((idx/step_size)+1))
        img, noise = forward_diffusion_sample(img, t, device)
        img = torch.clamp(img, -1.0, 1.0)
        img = img.cpu()
        plt.imshow(img.permute(1, 2, 0))

    plt.show()

    return

# display_forward_diffusion()


# U-NET - used for backward diffusion
# convlolutions, down and up sampling, residual connections
# similar to auto-encoder
# denoising score matching
# positional embeddings are used for the step in the sequence information (t)
# U-Net needs to predict the noise and subtract it from the image (to get image at noise step t-1)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

# upsaling downsampling convolutions?
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SelfAttention(nn.Module):

    def __init__(self, in_ch, size):
        super().__init__()
        self.channels = in_ch
        self.size = size
        self.attention = nn.MultiheadAttention(in_ch, 4, batch_first=True)
        self.ln = nn.LayerNorm([in_ch])
        self.seq = nn.Sequential(
            nn.LayerNorm([in_ch]),
            nn.Linear(in_ch, in_ch),
            nn.GELU(),
            nn.Linear(in_ch, in_ch))

    def forward(self, x):
        # try:
        #     x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        # except:
        #     x = x.reshape(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.attention(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.seq(attention_value) + attention_value
        out = attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        return out



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()

        self.time_emb_dim = 256
        self.device = 'cuda'

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        # check
        self.conv0 = DoubleConv(3, 64)

        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 512)
        self.sa3 = SelfAttention(512, 8)
        self.down4 = Down(512, 512)
        self.sa4 = SelfAttention(512, 4)

        self.conv1 = nn.Sequential(
            DoubleConv(512, 1024),
            DoubleConv(1024, 1024),
            DoubleConv(1024, 512)
        )

        self.up1 = Up(1024, 256)
        self.sa5 = SelfAttention(256, 8)
        self.up2 = Up(512, 128)
        self.sa6 = SelfAttention(128, 16)
        self.up3 = Up(256, 64)
        self.sa7 = SelfAttention(64, 32)
        self.up4 = Up(128, 64)
        self.sa8 = SelfAttention(64, 64)

        self.output = nn.Conv2d(64, 3, 1)


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, x, timestep):

        # Embedd time
        # t = self.time_mlp(timestep)
        t = timestep.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_emb_dim)

        # Initial conv
        x0 = self.conv0(x)
        # Unet

        x1 = self.down1(x0, t)
        x1 = self.sa1(x1)
        x2 = self.down2(x1, t)
        x2 = self.sa2(x2)
        x3 = self.down3(x2, t)
        x3 = self.sa3(x3)
        x4 = self.down4(x3, t)
        x4 = self.sa4(x4)

        x = self.conv1(x4)

        x = self.up1(x, x3, t)
        x = self.sa5(x)

        x = self.up2(x, x2, t)
        x = self.sa6(x)

        x = self.up3(x, x1, t)
        x = self.sa7(x)

        x = self.up4(x, x0, t)
        # x = self.sa8(x)

        x = self.output(x)

        return x

def get_loss(model, x0, t, device='cuda'):
    x_noisy, noise = forward_diffusion_sample(x0, t, device)
    x_noisy = torch.clamp(x_noisy, -1.0, 1.0)
    noise_pred = model(x_noisy, t)

    # TODO check L1 loss
    # l1_loss = F.l1_loss(noise, noise_pred)
    # loss = l1_loss(noise, noise_pred)
    mse = nn.MSELoss()
    loss = mse(noise, noise_pred)

    return loss


# uses the model to predict the noise, next denoise the image
# applies noise to this image, if not in the last step
@torch.no_grad()
def sample_timestep(x, t):
    model.eval()
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    # sqrt(1/alpha)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # call model
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        noise = torch.zeros_like(x)
    else:
        noise = torch.randn_like(x)

    # out = model_mean + torch.sqrt(posterior_variance_t) * noise

    alpha_t = get_index_from_list(alphas, t, x.shape)
    out = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alphas_cumprod_t) * model(x, t)) + torch.sqrt(betas_t) * noise

    # reddit fix
    # alpha_prev = get_index_from_list(alphas_prev, t, x.shape)
    # x0 = (x - sqrt_one_minus_alphas_cumprod_t * model(x, t)) / torch.sqrt(alpha_t)
    # out = torch.sqrt(alpha_prev) * x0 + torch.sqrt(posterior_variance_t) * noise

    model.train()

    return out


@torch.no_grad()
def sample_plot_image(img_size, device, epoch):

    # disable displaying of plots
    plt.ioff()
    # sample noise
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    n_images = 20
    step = int(T / n_images)

    idx = 1
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = torch.clamp(img, -1, 1)
        img = sample_timestep(img, t) 

        if i % step == 0:
            plt.subplot(2, int(n_images/2), idx)
            img_show = img.detach().cpu()
            img_show = torch.clamp(img_show, -1.0, 1.0)
            img_show = (img_show + 1) / 2
            img_show = (img_show * 255).type(torch.uint8)
            img_show = img_show[0].permute(1, 2, 0)
            img_show = img_show.cpu().numpy()
            plt.imshow(img_show)
            idx += 1

    plt.savefig(f'samples/epoch_{epoch}.png')
    plt.close('all')
    plt.ion()

    return

device = 'cuda:0'
model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)
model.to(device)

# add if exists
path_exist = os.path.exists(f"{os.getcwd()}\\model_state.pth")
if path_exist:
    model.load_state_dict(torch.load('model_state.pth'))

path_exist = os.path.exists(f"{os.getcwd()}\\epoch.txt")
if path_exist:
    with open('epoch.txt', 'r') as f:
        content = f.read()
    epoch_min_range = int(content)
else:
    epoch_min_range = 0

opt = Adam(model.parameters(), lr=0.0003)
epochs = 1000

for epoch in range(epoch_min_range, epoch_min_range+epochs):
    losses = np.zeros(0)
    start = time.time()
    # for step, batch in enumerate(dl):
    for batch_x, batch_y in dl:

        opt.zero_grad()

        t = torch.randint(0, T, (batch_size,), device=device).long()
        # loss = get_loss(model, batch[0], t, device)
        loss = get_loss(model, batch_x, t, device)
        loss.backward()
        opt.step()

        with torch.no_grad():
            losses = np.append(losses, loss.cpu().numpy())


    elapsed = time.time() - start
    print(f'epoch: {epoch}, loss: {np.mean(losses):.4f}'
          f' time: {elapsed/60:.1f} minutes')

    if epoch % 5 == 0:
        torch.save(model.state_dict(), 'model_state.pth')
        with open('epoch.txt', 'w') as f:
            f.write(str(epoch))

        sample_plot_image(img_size, device, epoch)


# increasing SA heads from 4 to 8 didnt change much
# adding last SA layer completely brakes compute time

# changes since commit:
# relu -> gelu in self attention
# time embeddings changed
# beta scheduler back to linear
# changed sampling

# output of the model is huuuuge