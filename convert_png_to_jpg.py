from PIL import Image 
import os 

path = r'D:\\car_images\\dataset1\\images_png\\'

for file in os.listdir(path): 
    if file.endswith(".png"): 
        img = Image.open(path + file).convert("RGB")
        file_name, file_ext = os.path.splitext(file)
        img.save(r'D:\\car_images\\dataset1\\images_jpg\\{}.jpg'.format(file_name))