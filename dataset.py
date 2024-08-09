from os import listdir
from os.path import isfile, join
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import warnings
warnings.filterwarnings("ignore")
torchvision.disable_beta_transforms_warning()
from torchvision import transforms
import json
from PIL import ImageColor, Image
from torchvision import transforms as tfms

def multi_hex2rgb(lst_hex):
    lst_rgb = []
    for hex in lst_hex:
        lst_rgb.append(ImageColor.getcolor(hex, "RGB"))
    return lst_rgb

def pattern_check(meta):
    if "spots" in meta.keys():
        return 1
    elif "pixelize" in meta.keys():
        return 2
    else:
        return 0
    
class CamouflageDataset(Dataset):

    def __init__(self, image_dir, meta_dir, image_size=700):
        self.images = [join(image_dir,f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
        self.metas = [join(meta_dir,f) for f in listdir(meta_dir) if isfile(join(meta_dir, f))]
        
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]),])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img = Image.open(self.images[idx])
        img = img.resize((512,512))
        img = tfms.ToTensor()(img)*2-1
        # img = self.transform(img)

        # Load metadata
        with open(self.metas[idx]) as f:
            meta = json.load(f)
        
        # Load color
        colors = multi_hex2rgb(meta['colors'])
        colors = torch.Tensor(colors)/255.0

        # Load pattern
        pattern_code = pattern_check(meta)
        
        return img, colors, pattern_code
