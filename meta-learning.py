from dataset import CamouflageDataset
import torch
from diffusers.utils import make_image_grid
from torchvision import transforms as tfms
from diffusers import AutoencoderKL
from diffusers import UNet2DModel
from PIL import Image
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import os
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import argparse

def train_loop(model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, total_epochs):
    grad_accumulation_steps = 2
    # Now you train the model
    for epoch in range(total_epochs):
        progress_bar = tqdm(train_dataloader, position=0, desc=f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0].cuda()
            cond = batch[2].cuda()
            
            clean_latents = img_to_latent(clean_images)
            # Sample noise to add to the images
            noise = torch.randn(clean_latents.shape, device=clean_latents.device)
            bs = clean_latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_latents.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

            # Predict the noise residual
            noise_pred = model(noisy_latents, timesteps, class_labels=cond, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)

            if (step + 1) % grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

        evaluate(epoch, noise_scheduler, model)    
        torch.save(model.state_dict(), os.path.join(args.output_dir, "ckpt_{epoch}.pth"))
                
def evaluate(epoch, noise_scheduler, model):
    device = 'cuda'
    x = torch.randn(8, 4, 64, 64).to(device)
    cond = torch.randint(low=0, high=3, size=(8,)).cuda()
    for i, t in tqdm(enumerate(noise_scheduler.timesteps), position=1, desc="Sampling"):
        model_input = noise_scheduler.scale_model_input(x, t)
        with torch.no_grad():
            noise_pred = model(model_input, t, class_labels=cond)["sample"]
        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    images = latent_to_img(x)
    image_grid = make_image_grid(images, rows=2, cols=4)

    # Save the images
    test_dir = "samples/cond"
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    
# Convert PIL images to latents
def img_to_latent(input_im):
    with torch.no_grad():
        latent = vae.encode(input_im) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

# Convert latents to PIL images
def latent_to_img(latent):
    with torch.no_grad():
        decoded = vae.decode(latent / 0.18215).sample
        image_tensor = (decoded + 1) / 2  # Rescale to [0, 1]
        image_tensor = torch.clamp(image_tensor, 0, 1)
        lst = []
        for i in range(image_tensor.shape[0]):
            lst.append(tfms.ToPILImage()(image_tensor[i]))
        return lst

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="path to the dataset folder", default="dataset")
parser.add_argument("--output-dir", help="path to the output folder", default="trained_models")

args = parser.parse_args()

dataset = CamouflageDataset(os.path.join(args.dataset, "images"), os.path.join(args.dataset, "metadata"))

batch_size = 8
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to('cuda')

# Unet model
model = UNet2DModel(
    sample_size=64,  # the target image resolution
    in_channels=4,  # the number of input channels
    out_channels=4,  # the number of output channels
    class_embed_type='timestep',
    num_class_embeds=3,
).cuda()

# Scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

total_epochs = 50

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(dataloader) * total_epochs),
)

train_loop(model, noise_scheduler, optimizer, dataloader, lr_scheduler, total_epochs)
