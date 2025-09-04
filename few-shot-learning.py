from diffusers import DDPMScheduler
from diffusers import UNet2DModel
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tfms
from tqdm import tqdm
from diffusers import AutoencoderKL
from diffusers.utils import make_image_grid
from os import listdir
from os.path import isfile, join
import numpy as np
import torch 
import torchvision.transforms as transforms 
from PIL import Image 
import random
import matplotlib.pyplot as plt
import pickle
import argparse

# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------

def generate_results(latent):    
    """Decode latent tensor back into list of PIL images."""
    with torch.no_grad():
        decoded = vae.decode(latent / 0.18215).sample  # Decode latent (divide by scale factor)
        image_tensor = (decoded + 1) / 2               # Rescale from [-1, 1] → [0, 1]
        image_tensor = torch.clamp(image_tensor, 0, 1) # Ensure valid pixel range
        lst = []
        for i in range(image_tensor.shape[0]):
            lst.append(tfms.ToPILImage()(image_tensor[i]))
        return lst

def latent_to_img(latent):
    """Decode latent to image tensor (not PIL)."""
    decoded = vae.decode(latent / 0.18215).sample
    image_tensor = (decoded + 1) / 2  # Rescale to [0, 1]
    image_tensor = torch.clamp(image_tensor, 0, 1)
    return image_tensor

def img_to_latent(input_im):
    """Encode an input image tensor into VAE latent space."""
    with torch.no_grad():
        latent = vae.encode(input_im)  # Encode to distribution
    return 0.18215 * latent.latent_dist.sample()  # Apply SD scaling factor

def latent_loss(latent, total_scenes, target_latents):
    """
    Compute average L1 loss between current latent and a set of target latents.
    Latent is repeated across all target scenes for comparison.
    """
    latent_repeat = latent.repeat(total_scenes, 1, 1, 1)
    error = torch.abs(latent_repeat - target_latents).mean()
    return error

if __name__ == "__main__":
    # ------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", help="pth file", default="trained-models/ckpt_49.pt")
    parser.add_argument("--input-folder", help="input folder", default="evaluation-datasets/custom-dataset/forest")
    parser.add_argument("--guidance-factor", help="guidance factor", default=40)
    parser.add_argument("--timesteps", help="number of timesteps/epochs", default=200)
    parser.add_argument("--output-file", help="output file name", default="output.png")

    args = parser.parse_args()

    # ------------------------------------------------------
    # Model & scheduler setup
    # ------------------------------------------------------

    # Diffusion noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.timesteps)

    # Define UNet denoising model (operates in latent space)
    model = UNet2DModel(
        sample_size=64,   # Latent resolution (64x64 for 512x512 images)
        in_channels=4,    # Latent channels
        out_channels=4,   # Output channels (predict noise in latent space)
        class_embed_type='timestep',
        num_class_embeds=3,  # Number of discrete condition classes
    ).cuda()

    # Load pretrained model weights
    model.load_state_dict(torch.load(args.pretrained))
    model.eval()

    # Load pretrained VAE for encoding/decoding latents
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to('cuda')
    vae.eval()

    # ------------------------------------------------------
    # Prepare target latents
    # ------------------------------------------------------

    target_path = args.input_folder
    target_files = [f for f in listdir(target_path) if isfile(join(target_path, f))]
    target_lst = []

    for file in target_files:
        # Open image
        target_image = Image.open(join(target_path, file))
        target_image = target_image.resize((500,500))  # Resize for consistency
        
        # Extract 10 random crops from each image
        for i in range(10):
            x1 = random.randrange(0, 400)
            y1 = random.randrange(0, 400)
            
            # Crop 100x100 patch
            visualize_img = target_image.crop((x1, y1, x1 + 100, y1 + 100))
            crop_img = visualize_img.resize((512,512))  # Resize patch to 512x512
            
            # Convert to tensor in [-1, 1]
            crop_img = tfms.ToTensor()(crop_img)*2-1
            crop_img = crop_img[:3,:,:]                 # Keep RGB only
            crop_img = crop_img.unsqueeze(0).cuda()     # Add batch dimension
            
            # Encode to latent
            target_latent = img_to_latent(crop_img).squeeze()
            target_lst.append(target_latent)

    # Stack all target latents
    target_latents = torch.stack(target_lst).cuda()
    total_scenes = len(target_latents)

    # ------------------------------------------------------
    # Guided sampling
    # ------------------------------------------------------

    # Scale factor for guidance loss
    guidance_loss_scale = args.guidance_factor
    device = "cuda"
    batch = 1

    # Start from random noise
    x = torch.randn(batch, 4, 64, 64).to(device)
    # Random conditioning label
    cond = torch.randint(low=0, high=3, size=(batch,)).cuda()

    # Iterative reverse diffusion
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        x = x.detach().requires_grad_()

        # Scale latent input for current timestep
        noisy_latents = noise_scheduler.scale_model_input(x, t)

        # Predict noise with UNet
        noise_pred = model(noisy_latents, t, class_labels=cond)['sample']

        # Predicted denoised latent x0
        x0 = noise_scheduler.step(noise_pred, t, x).pred_original_sample

        # Compute guidance loss with respect to target latents
        loss = latent_loss(x0, total_scenes, target_latents)*guidance_loss_scale
        
        if i % 10 == 0:
            print(i, "loss:", loss.item())

        # Compute gradient of loss w.r.t. x
        cond_grad = -torch.autograd.grad(loss, x)[0]

        # Update latent using guidance gradient
        x = x.detach() + cond_grad

        # Perform diffusion step (reverse process)
        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    # ------------------------------------------------------
    # Decode & save result
    # ------------------------------------------------------

    # Decode final latent to image
    generated = generate_results(x)
    camou = generated[0]
    # Save camouflage result
    camou.save(args.output_file)
