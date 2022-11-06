import os

# iterate trough folder
for filename in os.listdir('D:\car_images\merged_annotations'):
    path = 'D:\\car_images\\merged_annotations\\' + filename
    with open(path, "r") as f:
        lines = f.readlines()
    with open(path, "w") as f:
        for line in lines:
            if line.split(" ")[0] == "0":
                f.write(line)