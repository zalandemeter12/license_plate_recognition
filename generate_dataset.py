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


def synthesize_image(output_name, foreground_src, background_src, output_size, rot_angle, tilt_angle, scale, x_offset, y_offset, skew_angle, noise_bound, blur_radius, change_brightness=True):    
    #background = Image.open('high_res_images/TAA-262.jpg')
    background = Image.open(background_src)
    #resize background image
    background = background.resize(output_size)
    #img = Image.open('license_plate_images/TAA-262.jpg')
    img = Image.open(foreground_src)

    # scale image
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))

    width, height = img.size
    tilt_angle = tilt_angle / 180 * np.pi
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

    width, height = img.size
    m = skew_angle
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    img = img.transform((new_width, height), Image.AFFINE,
            (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)



    img = img.rotate(rot_angle, expand=True)
    

    # change brightness
    if change_brightness:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.2, 1.8))

    #img = Image.eval(img, lambda x: x * random.uniform(0.5, 1.5))

    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    
    x_offset = np.clip(x_offset, 0, background.size[0] - img.size[0])
    y_offset = np.clip(y_offset, 0, background.size[1] - img.size[1])

    background.paste(img, (x_offset, y_offset), img)

    # add noise
    background = np.array(background)
    background = background + np.random.normal(-noise_bound, noise_bound, background.shape)
    background = np.clip(background, 0, 255)
    background = Image.fromarray(background.astype('uint8'))

    if change_brightness:
        enhancer = ImageEnhance.Brightness(background)
        background = enhancer.enhance(random.uniform(0.5, 1.5))

    #background.show()

    #create synthesized_images folder if it doesn't exist
    if not os.path.exists('synthesized_images'):
        os.makedirs('synthesized_images')

    #save background filename: foreground file name, x_offset, y_offset, foreground width, foreground height
    new_filename = 'synthesized_images/' + str(output_name) + '_' + str(x_offset) + '_' + str(y_offset) + '_' + str(img.size[0]) + '_' + str(img.size[1])
    background.save(new_filename + '.jpg')

    # create a text file with the same name as the image file
    # the text file will contain the x_offset, y_offset, foreground width, foreground height
    f = open(new_filename + '.txt', 'w')
    f.write('0 ' + str(x_offset/background.size[0]) + ' ' + str(y_offset/background.size[1]) + ' ' + str(img.size[0]/background.size[0]) + ' ' + str(img.size[1]/background.size[1]))
    f.close()

    

def main():
    #shuffle files in background_images folder
    background_images = os.listdir('background_images')
    random.shuffle(background_images)

    # iterate through all images in the folder positive_images
    for index, filename in enumerate(os.listdir('positive_images')):
        if(index == 5):
            break
        background = random.choice(background_images)
        output_size = (800, 600)
        rot_angle = random.randint(-45, 45)
        tilt_angle = random.randint(-80, 80)
        scale = random.uniform(0.2, 3)
        x_offset = random.randint(0, output_size[0])
        y_offset = random.randint(0, output_size[1])
        skew_angle = random.uniform(-0.5, 0.5)
        noise_bound = random.randint(0, 30)
        blur_radius = random.uniform(0, 2.5)
        change_brightness = random.choice([True, False])

        synthesize_image(index, 'positive_images/' + filename, 'background_images/' + background, output_size, rot_angle, tilt_angle, scale, x_offset, y_offset, skew_angle, noise_bound, blur_radius, True)

if __name__ == '__main__':
    main()