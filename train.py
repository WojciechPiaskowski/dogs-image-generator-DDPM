# imports
import argparse
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from matplotlib import pyplot as plt
from PIL import Image
import os
from DDPM import Unet, EMA
import utility_f as uti

if __name__ == '__main__':

    # training arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--examples', type=bool, default=False)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--save_frequency', type=int, default=10)

    args = parser.parse_args()
    img_size = 64

    # beta scheduler
    betas = uti.linear_beta_schedule(timesteps=args.T)
    # betas = uti.cosine_beta_schedule(timesteps=args.T)

    # remove corrupted image files
    rt_path = f'{os.getcwd()}\\data\\dogs'
    for subdir, dirs, files in os.walk(rt_path):
        for file in files:
            path = os.path.join(subdir, file)
            try:
                img = Image.open(path)
            except:
                os.remove(path)

    # load the images to dataloader
    dl = uti.load_transformed_dataset(path='data/dogs', img_size=img_size, batch_size=args.batch_size)

    # show sample images
    if args.examples:
        uti.show_images(dl, num_samples=args.samples, display=True, title='sample_images')

    # pre-calculates values used to get noisy image at specific timestep (cumulated product of alpha)
    # as well as other values used for prediction (generation)
    alphas = 1.0 - betas
    alphas_prev = F.pad(alphas[:-1], (1, 0), value=1.0)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # takes an image and a timestep, returns noisy image (and noise itself) at that timestep
    def forward_diffusion_sample(x0, t, device=args.device):

        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = uti.get_index_from_list(sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = uti.get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x0.shape)

        # mean + variance
        mean = sqrt_alphas_cumprod_t.to(device) * x0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        noisy_img = mean + variance

        noise = noise.to(device)

        return noisy_img, noise

    # simulate forward diffusion over single image
    def display_forward_diffusion(device=args.device):

        img = next(iter(dl))[0][0]
        img = (img + 1) / 2

        fig = plt.figure(figsize=(15, 15))
        plt.axis('off')
        n_images = 10
        step_size = int(args.T / n_images)

        for idx in range(0, args.T, step_size):
            t = torch.tensor([idx]).type(torch.int64)
            fig.add_subplot(1, n_images + 1, int((idx / step_size) + 1))
            img, noise = forward_diffusion_sample(img, t, device)
            img = torch.clamp(img, 0.0, 1.0)
            img = img.cpu()
            plt.imshow(img.permute(1, 2, 0))

        plt.show()

        return


    if args.examples:
        display_forward_diffusion()

    # function create a noisy version of the image based image (x0) and timestep (t)
    # next it makes noise prediction using the model
    # then it calculate MSE loss based on noise and noise prediction
    def get_loss(model, x0, t, device=args.device):

        x_noisy, noise = forward_diffusion_sample(x0, t, device)
        x_noisy = torch.clamp(x_noisy, -1.0, 1.0)
        noise_pred = model(x_noisy, t)

        mse = nn.MSELoss()
        loss = mse(noise, noise_pred)

        return loss

    # uses the model to predict the noise, next denoises the image
    @torch.no_grad()
    def sample_timestep(x, t, model):

        # set model to evaluation mode
        model.eval()

        betas_t = uti.get_index_from_list(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = uti.get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)

        # noise is 0 if timestep is 0, else its normally distributed
        if t == 0:
            noise = torch.zeros_like(x)
        else:
            noise = torch.randn_like(x)

        alpha_t = uti.get_index_from_list(alphas, t, x.shape)
        noise_pred = model(x, t)

        out = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alphas_cumprod_t) * noise_pred) + \
              torch.sqrt(betas_t) * noise

        # set model back to training mode
        model.train()

        return out

    # function to generate and save images
    @torch.no_grad()
    def sample_plot_image(img_size, n_images, device, model, title):

        # disable displaying of plots
        plt.ioff()
        # sample noise as initial image for generation
        img = torch.randn((1, 3, img_size, img_size), device=device)
        plt.figure(figsize=(15, 15))
        plt.axis('off')

        idx = 1
        rows = int(n_images / 10) + 1
        cols = int(n_images / rows) + 1

        fig = plt.figure(figsize=(15, 15))
        for j in range(n_images):
            for i in range(0, args.T)[::-1]:
                t = torch.full((1,), i, device=device, dtype=torch.long)
                img = torch.clamp(img, -1, 1)
                # denoise the image using the model
                img = sample_timestep(img, t, model)

            fig.add_subplot(rows, cols, idx)
            img_show = img.detach().cpu()
            img_show = torch.clamp(img_show, -1.0, 1.0)
            img_show = (img_show + 1) / 2
            img_show = (img_show * 255).type(torch.uint8)
            img_show = img_show[0].permute(1, 2, 0)
            img_show = img_show.cpu().numpy()

            plt.imshow(img_show)
            idx += 1

        # save generated image
        plt.savefig(f'samples/{title}.png')

        plt.close('all')
        plt.ion()

        return


    # initiate model, train on GPU
    model = Unet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    # 93 417 859 parameters
    print(model)
    model.to(args.device)

    # initiate EMA (exponential moving average) model
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # load model weights, if they already exists
    path_exist = os.path.exists(f"{os.getcwd()}\\model_state.pth")
    if path_exist:
        model.load_state_dict(torch.load('model_state.pth'))

    # read last saved epoch, if the file already exist
    path_exist = os.path.exists(f"{os.getcwd()}\\epoch.txt")
    if path_exist:
        with open('epoch.txt', 'r') as f:
            content = f.read()
        epoch_min_range = int(content)
    else:
        epoch_min_range = 0

    # initiate adam optimizer, learning rate and number of epochs
    opt = Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs

    # training loop
    for epoch in range(epoch_min_range, epoch_min_range + epochs):

        losses = np.zeros(0)
        start = time.time()
        for batch_x, batch_y in dl:
            # reset the gradients
            opt.zero_grad()

            # set a random noise timestep
            t = torch.randint(0, args.T, (args.batch_size,), device=args.device).long()

            # calculate loss and update the weights in both base model and EMA model
            loss = get_loss(model, batch_x, t, args.device)
            loss.backward()
            opt.step()
            ema.step_ema(ema_model, model)

            # append batch loss to epoch losses array
            with torch.no_grad():
                losses = np.append(losses, loss.cpu().numpy())

        # print out time elapsed, epoch and average loss across batches
        elapsed = time.time() - start
        print(f'epoch: {epoch}, loss: {np.mean(losses):.4f}'
              f' time: {elapsed / 60:.1f} minutes')

        # every 10th epoch save parameters, epoch number and generate sample images
        if epoch % args.save_frequency == 0:
            torch.save(model.state_dict(), 'model_state.pth')
            torch.save(ema_model.state_dict(), 'ema_model_state.pth')
            with open('epoch.txt', 'w') as f:
                f.write(str(epoch))

            sample_plot_image(n_images=args.samples, img_size=img_size, device=args.device, model=model,
                              title=f'epoch_{epoch}')
            sample_plot_image(n_images=args.samples, img_size=img_size, device=args.device, model=model,
                              title=f'ema_epoch_{epoch}')
