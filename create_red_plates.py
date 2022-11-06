import cv2
import imutils
import numpy as np 
import os


# change every black pixel to red on image
# iterate through old_hu_license_plates folder

for filename in os.listdir('data/old_hu_license_plates'):
    img_rgb = cv2.imread("data/old_hu_license_plates/" + filename) 

    Conv_hsv_Gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)

    indices = np.where(mask==255)

    img_rgb[indices[0], indices[1], :] = [31, 31, 165]
    
    # save image
    cv2.imwrite("data/synthetic_red_plates/" + filename + "_red.jpg", img_rgb)
