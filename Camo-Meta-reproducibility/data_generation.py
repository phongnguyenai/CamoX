from os import listdir
from os.path import isfile, join
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import random
import json
from color_embedding import KMeansCUDA
import camogen

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))

def multi_rgb2hex(lst_rgb):
    lst_hex = []
    for rgb in lst_rgb:
        lst_hex.append(rgb2hex(rgb[0], rgb[1], rgb[2]))
    return lst_hex

def generate_params(colors):
    seed = random.randint(0,2)
    
    if seed == 0:
        parameters = {'width': 700, 
                  'height': 700, 
                  'polygon_size': random.randint(200,400), 
                  'color_bleed': random.randint(0,2),
                  'colors': colors,
                  'max_depth': 15,
                  }
    
    elif seed == 1:
        parameters = {'width': 700, 
                  'height': 700, 
                  'polygon_size': random.randint(200,400), 
                  'color_bleed': random.randint(0,5),
                  'colors': colors,
                  'max_depth': 15,
                  
                  'spots': {
                      'amount': 5000,
                      'radius': {'min': 7, 'max': 18},
                      'sampling_variation': 10
                  }
                  }
        
    elif seed == 2:
        parameters = {'width': 700, 
                  'height': 700, 
                  'polygon_size': random.randint(200,400), 
                  'color_bleed': random.randint(0,1),
                  'colors': colors,
                  'max_depth': 15,
    
    
                  'pixelize': {
                      'percentage': random.randint(90,100)/100,
                      'sampling_variation': 50,
                      'density': {'x': 50, 'y': 50}
                  }
                  }
        
    return parameters

def save_meta(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    scene_dir = "scenary-data"
    scene_files = [join(scene_dir,f) for f in listdir(scene_dir) if isfile(join(scene_dir, f))]
    
    num_colors = 10
    kmean = KMeansCUDA(n_clusters=num_colors)
    
    count = 0
    for i in range(3):
        for file in scene_files:
            ori_img = cv2.imread(file)
            img = cv2.resize(ori_img, (256,256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.Tensor(img)
            
            points = img.view(-1,3).cuda()
            rgb_colors = kmean.fit(points)
            rgb_colors = rgb_colors.int()
            rgb_colors = rgb_colors.to('cpu').numpy().tolist()
            hex_colors = multi_rgb2hex(rgb_colors)
        
            random.shuffle(hex_colors)
            dominant_colors = hex_colors[:4]
            params = generate_params(dominant_colors)
            image = camogen.generate(params)
            save_meta(params, f"dataset/metadata/{count}.json")
            image.save(f"dataset/images/{count}.png")
            count+=1