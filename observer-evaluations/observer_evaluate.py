from PIL import Image,ImageDraw
import pickle
import random
import os
import cv2
import time
import json
import numpy
from os import listdir
from os.path import isfile, join

def draw_circle(event,x,y,flags,param):
    global check, x_gt, y_gt, size, img_size
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if x_gt < x < x_gt+size:
            if y_gt < y < y_gt+size:
                if check == 0:
                    result[scene] += time.time()-start
                    result['total'] += time.time()-start
                    cv2.rectangle(img, (max(0, x_gt-10), max(0, y_gt-10)), (min(x_gt+size+10, img_size), min(y_gt+size+10, img_size)), (0,0,255), 2)
                    cv2.imshow("HumanTrial", img)
                    print(f"Done, please press 'N'")
                    check = 1
                    
scenes = ["mangroves", "beach", "forest", "rocky-mountain", "snow", "everglades", "bush", "desert"]
model_name = input("Model name: ")
name = input("Participant name: ")
result = {}
for scene in scenes:
    result[scene] = 0
    
result['total'] = 0
i = 0

while i < 25:
    scene = random.choice(scenes)
    size = random.choice([60,80,100])
    img_size = 800

    path = "target"
    list_animals = [f for f in listdir(path) if isfile(join(path, f))]
    target_choice = random.choice(list_animals)
    scene_path = f"../test/{scene}"
    
    with open(f'../test/split/{scene}.pickle', 'rb') as handle:
        b = pickle.load(handle)
    
    target_files = b['train'] + b['test']
    
    file = random.choice(target_files)
    
    img = Image.open(f"{scene_path}/{file}")
    if scene == "everglades":
        img = img.rotate(180)

    x_gt = random.randint(0, img_size-size)
    y_gt = random.randint(0, img_size-size)

    camou = Image.open(f"../results/{model_name}/{model_name}_{scene}.png")
    target = Image.open(f"target/{target_choice}").convert("RGBA")

    # Resize the camouflage image to match the bird image size
    camou_resized = camou.resize(target.size)
    
    # Extract the alpha channel from the bird image to use as a mask
    mask = target.split()[3]  # Get the alpha channel
    
    # Create a new image by applying the camouflage pattern to the bird shape
    camouflaged_target = Image.composite(camou_resized, target, mask)

    camouflaged_target = camouflaged_target.resize((size,size))   
    target_mask = camouflaged_target.split()[3]

    camou_img = img.resize((img_size,img_size))
    
    camou_img.paste(camouflaged_target, (x_gt,y_gt), target_mask)
    
    img = numpy.array(camou_img)
    target = target.resize((img_size,img_size))
    target_cv = numpy.array(target)[:,:,:3]

    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()
    target_cv = target_cv[:, :, ::-1].copy()
    
    img = numpy.concatenate((img, target_cv), axis=1)
    cv2.imshow("HumanTrial", img)
    check = 0
    cv2.setMouseCallback("HumanTrial", draw_circle)
    start = time.time()
    k = cv2.waitKey(0)
    if k == ord("n"):
        if check == 1:
            i += 1
            cv2.destroyAllWindows()
        else:
            if i > 0:
                i -= 1
            print(i)
            cv2.destroyAllWindows()
            

result['total'] = result['total']
with open(f'results/{name}_{model_name}.json', 'w') as f:
    json.dump(result, f)





