import os
from PIL import Image


def resize_images(input_dir, output_dir,height=300, width=300):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            image = Image.open(input_path)
            resized_image = image.resize((width, height))
            resized_image.save(output_path)


input_dir = r"input_path"

output_dir = r"output_path"


resize_images(input_dir, output_dir, height=600, width=600)