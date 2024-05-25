import os
import cv2
import albumentations as A
import random
import numpy as np

transform = A.Compose([
    A.VerticalFlip(p=0.5),  
    A.Rotate(limit=45),  
    A.Blur(blur_limit=(3, 7)),  
    A.PadIfNeeded(min_height=500, min_width=500, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]), 
])

input_dir = "input_path" 

output_dir = "output path"

os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]

max_augmentations_per_image = 3         # -> change as necessery 


for filename in image_files:
    image = cv2.imread(os.path.join(input_dir, filename))
    num_augmentations = random.randint(1, max_augmentations_per_image)
    for _ in range(num_augmentations):
        augmented = transform(image=image)
        augmented_image = augmented['image']
        output_path = os.path.join(output_dir, f"augmented_{filename[:-4]}_{_}.jpg")  
        cv2.imwrite(output_path, augmented_image)
