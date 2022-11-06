import os
import shutil
PATH = "D:\\car_images\\dataset1\\annotations\\"

def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)

            assert False

annotations = [os.path.join(PATH, x) for x in os.listdir(PATH) if x[-3:] == "txt"]
# Move the splits into their folders
move_files_to_folder(annotations, 'D:\\car_images\\dataset1\\annotations_txt')