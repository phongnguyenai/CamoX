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

def generate_results(latent):    
    with torch.no_grad():
        decoded = vae.decode(latent / 0.18215).sample
        image_tensor = (decoded + 1) / 2  # Rescale to [0, 1]
        image_tensor = torch.clamp(image_tensor, 0, 1)
        lst = []
        for i in range(image_tensor.shape[0]):
            lst.append(tfms.ToPILImage()(image_tensor[i]))
        return lst

# Convert latents to PIL images
def latent_to_img(latent):
    decoded = vae.decode(latent / 0.18215).sample
    image_tensor = (decoded + 1) / 2  # Rescale to [0, 1]
    image_tensor = torch.clamp(image_tensor, 0, 1)
    return image_tensor

# Convert PIL images to latents
def img_to_latent(input_im):
    with torch.no_grad():
        latent = vae.encode(input_im) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latent_loss(latent, total_scenes, target_latents):
    latent_repeat = latent.repeat(total_scenes, 1, 1, 1)
    error = torch.abs(latent_repeat - target_latents).mean()
    return error


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", help="pth file", default="trained-models/ckpt_49.pt")
parser.add_argument("--input-folder", help="input folder", default="evaluation-datasets/custom-dataset/forest")
parser.add_argument("--guidance-factor", help="guidance factor", default=40)
parser.add_argument("--timesteps", help="number of timesteps/epochs", default=200)
parser.add_argument("--output-file", help="output file name", default="output.png")

args = parser.parse_args()

noise_scheduler = DDPMScheduler(num_train_timesteps=args.timesteps)

# Unet model
model = UNet2DModel(
    sample_size=64,  # the target image resolution
    in_channels=4,  # the number of input channels
    out_channels=4,  # the number of output channels
    class_embed_type='timestep',
    num_class_embeds=3,
).cuda()

model.load_state_dict(torch.load(args.pretrained))
model.eval()

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to('cuda')
vae.eval()

target_path = args.input_folder
target_files = [f for f in listdir(target_path) if isfile(join(target_path, f))]
target_lst = []

for file in target_files:
    # target_image = Image.open(f"{target_path}/{file}")
    target_image = Image.open(join(target_path, file))
    target_image = target_image.resize((500,500))
    
    for i in range(10):
        x1 = random.randrange(0, 400)
        y1 = random.randrange(0, 400)
        visualize_img = target_image.crop((x1, y1, x1 + 100, y1 + 100))
        crop_img = visualize_img.resize((512,512))
     
        crop_img = tfms.ToTensor()(crop_img)*2-1
        crop_img = crop_img[:3,:,:]
        crop_img = crop_img.unsqueeze(0).cuda()
        target_latent = img_to_latent(crop_img).squeeze()
        
        target_lst.append(target_latent)

target_latents = torch.stack(target_lst).cuda()
total_scenes = len(target_latents)

# The guidance scale determines the strength of the effect
guidance_loss_scale = args.guidance_factor
device = "cuda"
batch = 1

x = torch.randn(batch, 4, 64, 64).to(device)
cond = torch.randint(low=0, high=3, size=(batch,)).cuda()

for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
    x = x.detach().requires_grad_()

    # Prepare the model input
    noisy_latents = noise_scheduler.scale_model_input(x, t)

                                            
    # Predict the noise residual
    noise_pred = model(noisy_latents, t, class_labels=cond)['sample']

    # Get the predicted x0
    x0 = noise_scheduler.step(noise_pred, t, x).pred_original_sample

    # kl_loss = kl_divergence_loss(images, target_image)*guidance_loss_scale
    loss = latent_loss(x0, total_scenes, target_latents)*guidance_loss_scale
    
    if i % 10 == 0:
        print(i, "loss:", loss.item())

    # Get gradient
    cond_grad = -torch.autograd.grad(loss, x)[0]

    # Modify x based on this gradient
    x = x.detach() + cond_grad

    # Now step with scheduler
    x = noise_scheduler.step(noise_pred, t, x).prev_sample

# Camouflage Detection
generated = generate_results(x)
camou = generated[0]
camou.save(args.output_file)
