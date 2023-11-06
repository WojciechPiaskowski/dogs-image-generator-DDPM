import torch
import torch.nn as nn
import torch.nn.functional as F


# exponential moving average class, that copies the model and smooths the training
class EMA:
    def __init__(self, beta):
        self.beta = beta

    # updates ema model weights based previous ema model weights and base model (updated) weights
    def step_ema(self, ema_model, model):
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_param.data, current_param.data
            ema_param.data = old_weight * self.beta + (1 - self.beta) * new_weight


# main building block in upscaling and downscaling layers
# double convolution layers with normalization
# can take residual connections
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


# Downsample block, lowers the resolution with maxpooling, increases number of feature maps
# also embeds the time dimension with a linear unit, merges it with double conv output
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


# Upsample block
# uses nn.Upsample to make the feature maps / images larger using weighted averages of nearest neighboring pixels
# upsamples only the normal input, the residual input is then merged before going into convolution layers
# lastly it is joined with separately processed time embedding
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
                out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# self attention layer
# utilizes transformer's self-attention where query, key and value are all normalized copies of the input
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
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.attention(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.seq(attention_value) + attention_value
        out = attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        return out


# DDPM Unet model class utilizing upsampling, downsampling and self attention blocks
class Unet(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_emb_dim = 256
        self.device = 'cuda'

        # Initial projection
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

    # function used to encode time dimension in a given embedding size (256 used)
    # sin / cosine is used to embed integer values in a continuous space
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    # forward propagation resulting in noise prediction,
    # which is size 16 (batch size) x 3 (channels) x 64 x 64 (width and height)
    def forward(self, x, timestep):
        # time embedding
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

        # final self attention layer removed, as it seems to slow the process incredibly on my machine
        # not sure exactly why that is
        # x = self.sa8(x)

        x = self.output(x)

        return x
