import cv2
import imutils
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import math
import random
import os

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def synthesize_image(output_name, foreground_src, background_src, output_size, rot_angle_bounds, tilt_angle_bounds, scale_bounds, skew_angle_bounds, noise_bounds, blur_radius_bounds, change_brightness=True):    
    #background = Image.open('high_res_images/TAA-262.jpg')
    background = Image.open(background_src)
    #resize background image
    background = background.resize(output_size)
    #img = Image.open('license_plate_images/TAA-262.jpg')
    new_filename = 'data/synthesized_images/' + str(output_name) + '_' + str(len(foreground_src))
    f = open(new_filename + '.txt', 'w')
    foreground_images = []
    for img_src in foreground_src:        
        x_offset = random.randint(0, output_size[0])
        y_offset = random.randint(0, output_size[1])
        img = Image.open(img_src).convert("RGBA")

        # scale image
        scale = random.uniform(scale_bounds[0], scale_bounds[1])
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        if random.choice([True, False]):
            width, height = img.size
            tilt_angle = random.randint(tilt_angle_bounds[0], tilt_angle_bounds[1]) / 180 * np.pi
            new_width = int(math.cos(tilt_angle) * width)
            if tilt_angle > 0:
                coeffs = find_coeffs(        
                        [(0, 0), (new_width, height*0.2), (new_width, height*0.8), (0, height)],
                        [(0, 0), (width, 0), (width, height), (0, height)])
            elif tilt_angle < 0:
                coeffs = find_coeffs(        
                        [(0, height*0.2), (new_width, 0), (new_width, height), (0, height*0.8)],
                        [(0, 0), (width, 0), (width, height), (0, height)])
            else:
                coeffs = find_coeffs(        
                        [(0, 0), (width, 0), (width, height), (0, height)],
                        [(0, 0), (width, 0), (width, height), (0, height)])
            img = img.transform((new_width, height), Image.PERSPECTIVE, coeffs,
                    Image.BICUBIC)
        
        if random.choice([True, False]):
            width, height = img.size
            m = random.uniform(skew_angle_bounds[0], skew_angle_bounds[1])
            xshift = abs(m) * width
            new_width = width + int(round(xshift))
            img = img.transform((new_width, height), Image.AFFINE,
                    (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)



        img = img.rotate(random.randint(rot_angle_bounds[0], rot_angle_bounds[1]), expand=True)
        

        # change brightness
        if change_brightness:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.2, 1.8))

        #img = Image.eval(img, lambda x: x * random.uniform(0.5, 1.5))

        img = img.filter(ImageFilter.GaussianBlur(random.uniform(blur_radius_bounds[0], blur_radius_bounds[1])))
        
        x_offset = np.clip(x_offset, 0, background.size[0] - img.size[0])
        y_offset = np.clip(y_offset, 0, background.size[1] - img.size[1])

        background.paste(img, (x_offset, y_offset), img)
        
        f.write('0 ' + str((x_offset + img.size[0]/2)/background.size[0]) + ' ' + str((y_offset+ img.size[1]/2)/background.size[1]) + ' ' + str(img.size[0]/background.size[0]) + ' ' + str(img.size[1]/background.size[1]) + '\n')

    # add noise
    background = np.array(background)
    noise_bound = random.randint(noise_bounds[0], noise_bounds[1])
    background = background + np.random.normal(-noise_bound, noise_bound, background.shape)
    background = np.clip(background, 0, 255)
    background = Image.fromarray(background.astype('uint8'))

    if change_brightness:
        enhancer = ImageEnhance.Brightness(background)
        background = enhancer.enhance(random.uniform(0.5, 1.5))

    #background.show()

    #create synthesized_images folder if it doesn't exist
    if not os.path.exists('data/synthesized_images'):
        os.makedirs('data/synthesized_images')

    #save background filename: foreground file name, x_offset, y_offset, foreground width, foreground height    
    background.save(new_filename + '.jpg')

    # create a text file with the same name as the image file
    # the text file will contain the x_offset, y_offset, foreground width, foreground height
        
    f.close()

    
POSITIVE_PATH = 'data/positive_images/'
BACKGROUND_PATH = 'D:\\background_images\\'

def main():
    #shuffle files in background_images folder
    background_images = os.listdir(BACKGROUND_PATH)
    random.shuffle(background_images)
    plates = os.listdir('data/positive_images/')
    # iterate through all images in the folder positive_images
    for index, filename in enumerate(plates):
        #if(index == 5):
        #    break

        num_of_images = random.randint(1, 3)
        choosen_plates = []
        choosen_plates.extend(random.choices(plates, k = num_of_images - 1))
        choosen_plates.append(filename)

        for i in range(len(choosen_plates)):
            choosen_plates[i] = 'data/positive_images/' + choosen_plates[i]

        background = random.choice(background_images)
        output_size = (1070, 800)
        rot_angle_bounds = (-45, 45)
        tilt_angle_bounds = (-60, 60)
        scale_bounds = (0.2, 2)
        
        skew_angle_bounds = (-0.5, 0.5)
        noise_bounds = (0, 30)
        blur_radius_bounds = (0, 2)
        change_brightness = random.choice([True, False])

        synthesize_image(index, choosen_plates, BACKGROUND_PATH + background, output_size, rot_angle_bounds, tilt_angle_bounds, 
                        scale_bounds, skew_angle_bounds, noise_bounds, blur_radius_bounds, True)

if __name__ == '__main__':
    main()