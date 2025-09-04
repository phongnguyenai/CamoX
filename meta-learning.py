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
    grad_accumulation_steps = 2  # Number of steps to accumulate gradients before optimizer step
    # Main training loop
    for epoch in range(total_epochs):
        progress_bar = tqdm(train_dataloader, position=0, desc=f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Extract input images and conditioning labels from batch
            clean_images = batch[0].cuda()  # Input images
            cond = batch[2].cuda()          # Conditioning labels
            
            # Encode images into latent space using VAE
            clean_latents = img_to_latent(clean_images)
            
            # Sample random Gaussian noise
            noise = torch.randn(clean_latents.shape, device=clean_latents.device)
            bs = clean_latents.shape[0]

            # Sample random timesteps for each sample in the batch
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_latents.device,
                dtype=torch.int64
            )

            # Add noise to latents (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

            # Predict the added noise using the UNet model
            noise_pred = model(noisy_latents, timesteps, class_labels=cond, return_dict=False)[0]
            
            # Compute mean squared error loss between predicted and actual noise
            loss = F.mse_loss(noise_pred, noise)
            
            # Backpropagation (accumulates gradients)
            loss.backward(loss)

            # Gradient accumulation step
            if (step + 1) % grad_accumulation_steps == 0:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                # Update learning rate scheduler
                lr_scheduler.step()

            # Update progress bar with loss and learning rate
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

        # Evaluate model at the end of each epoch
        evaluate(epoch, noise_scheduler, model)    

        # Save checkpoint after each epoch
        torch.save(model.state_dict(), os.path.join(args.output_dir, "ckpt_{epoch}.pth"))
                

def evaluate(epoch, noise_scheduler, model):
    device = 'cuda'
    # Start from random noise in latent space
    x = torch.randn(8, 4, 64, 64).to(device)
    # Random conditioning labels
    cond = torch.randint(low=0, high=3, size=(8,)).cuda()
    
    # Sampling loop (reverse diffusion process)
    for i, t in tqdm(enumerate(noise_scheduler.timesteps), position=1, desc="Sampling"):
        # Scale inputs for UNet
        model_input = noise_scheduler.scale_model_input(x, t)
        with torch.no_grad():
            # Predict noise at this timestep
            noise_pred = model(model_input, t, class_labels=cond)["sample"]
        # Update sample with scheduler step
        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    # Decode final latent to image space
    images = latent_to_img(x)
    # Arrange images in a grid
    image_grid = make_image_grid(images, rows=2, cols=4)

    # Save the grid of generated images
    test_dir = "samples/cond"
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    

# Convert PIL images to latent space
def img_to_latent(input_im):
    with torch.no_grad():
        latent = vae.encode(input_im) # Encode image to latent distribution
    # Scale latent (Stable Diffusion VAE uses factor 0.18215)
    return 0.18215 * latent.latent_dist.sample()


# Convert latent space back to PIL images
def latent_to_img(latent):
    with torch.no_grad():
        decoded = vae.decode(latent / 0.18215).sample  # Decode latent
        image_tensor = (decoded + 1) / 2  # Rescale to [0, 1]
        image_tensor = torch.clamp(image_tensor, 0, 1)  # Clamp values
        
        lst = []
        for i in range(image_tensor.shape[0]):
            # Convert tensor to PIL image
            lst.append(tfms.ToPILImage()(image_tensor[i]))
        return lst

if __name__ == "__main__":
    # -------------------------------
    # Argument parsing
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="path to the dataset folder", default="dataset")
    parser.add_argument("--output-dir", help="path to the output folder", default="trained_models")

    args = parser.parse_args()

    # Create dataset and dataloader
    dataset = CamouflageDataset(os.path.join(args.dataset, "images"), os.path.join(args.dataset, "metadata"))
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load pretrained VAE from Stable Diffusion
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to('cuda')

    # Define UNet model (core denoising network)
    model = UNet2DModel(
        sample_size=64,  # Target latent resolution
        in_channels=4,   # Input channels (latent space channels)
        out_channels=4,  # Output channels (predict noise)
        class_embed_type='timestep',
        num_class_embeds=3,  # Number of conditioning classes
    ).cuda()

    # Diffusion noise scheduler (forward/backward process)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Optimizer for training UNet
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    total_epochs = 50  # Number of training epochs

    # Learning rate scheduler (cosine with warmup)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(dataloader) * total_epochs),
    )

    # Start training
    train_loop(model, noise_scheduler, optimizer, dataloader, lr_scheduler, total_epochs)
