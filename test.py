import numpy as np
import cv2
import imutils
from PIL import Image, ImageEnhance
import paddleocr
import os
import random
import torch
import shutil
import time


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

CASCADE_PATH = 'cascade.xml'
RESIZE_HEIGHT = 600
CONF_THRESH = 0.5

# Initiate cascade classifer.
#plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best_s_200.pt', device='cpu')  
model.eval()

# Initiate OCR
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log = False, max_batch_size = 20, total_process_num = 12, use_mp=True) # need to run only once to download and load model into memory
image_dir = r'data/high_res_images/'
image_list = os.listdir(image_dir)
#random.shuffle(image_list)
#for z in range(1, 11):
correct_detections = 0
missed_images = []
times = []
for idx, original_filename in enumerate(image_list):
	tic = time.perf_counter()
	#print(original_filename)
	filename = image_dir + original_filename
	#img = cv2.imread('license_plate_' + str(z) + '.jpg')
	img = cv2.imread(filename)	
	#img  = imutils.resize(img, height = RESIZE_HEIGHT)

	#img_orig = img.copy()

	rotations = [0]
	# add list to rotations
	#for i in range(-46, 50, 23):
	#	if i == 0:
	#		continue
	#	rotations.append(i)
	
	# Detect plates in img
	plate_found = False
	plate_options = []
	for rotation in rotations:
		if plate_found:
			break
		#img = imutils.rotate_bound(img_orig , rotation)
		# Filter image to grayscale
		#img  = imutils.resize(img, height = RESIZE_HEIGHT)
		#gray_og = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)		
		#gray = cv2.equalizeHist(img)
		# show gray image and original image next to each other
		#cv2.imshow('img', img)
		#cv2.imshow('img', gray_og)
		#cv2.waitKey(0)

		#plates = plate_cascade.detectMultiScale(image=gray, scaleFactor=1.2, minNeighbors=20, minSize=(1,1), maxSize=(300,300))
		plates = model(img, size=RESIZE_HEIGHT)
		count = 0
		#print("Found %d plate candidates:" %len(plates))
		plates = plates.pandas().xyxy[0]
		#print(plates)
		#for (x,y,w,h) in plates:
		for index, row in plates.iterrows():
			#try text recognition
			x = int(row['xmin'])
			y = int(row['ymin'])
			w = int(row['xmax']) - x
			h = int(row['ymax']) - y

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)		
			#gray = cv2.equalizeHist(gray)

			roi = gray[y:y+h, x:x+w]
			# get roi size
			roi_height, roi_width = roi.shape

			scale = 100 / roi_height
			roi = cv2.resize(roi,dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
			enhancer = ImageEnhance.Contrast(Image.fromarray(roi))
			roi = enhancer.enhance(10)

			roi = np.array(roi)

			#cv2.imshow('img', roi)
			#pressed = cv2.waitKey(0)
			
			kernel = np.ones((3, 3), np.uint8)
  
            # Using cv2.erode() method 
			roi = cv2.erode(roi, kernel)

			#dilation
			kernel = np.ones((3, 3), np.uint8)
			roi = cv2.dilate(roi, kernel, iterations=1)

			#cv2.imshow('img', roi)
			#pressed = cv2.waitKey(0)	
						
			roi = cv2.resize(roi,(roi_width, roi_height))

			thresh = cv2.threshold(roi, 253, 255, cv2.THRESH_BINARY)[1]			
			contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			
			contours = contours[0] if len(contours) == 2 else contours[1]
			'''
			biggest_area = 0
			biggest_contour = 0
			for c in contours:
				area = cv2.contourArea(c)
				if area > biggest_area:
					biggest_area = area
					biggest_contour = c
			try:
				xx,yy,ww,hh = cv2.boundingRect(biggest_contour)
				#iterate over the roi and set to black all the pixels that are not in the bounding rectangle
				for i in range(0, roi.shape[0]):
					for j in range(0, roi.shape[1]):
						if i < yy or i > yy+hh or j < xx or j > xx+ww:
							roi[i][j] = 0
			except:
				continue
			'''


			data = ""
			conf = 0

			result = ocr.ocr(roi, cls=True)
			
			if len(result) < 1:
				continue
			for line in result:
				if len(line) < 1:
					continue
				for l in line:					
					data_cand = l[1][0]
					conf_cand = l[1][1]
					data = data + data_cand
					conf += conf_cand
				#print(data)
			conf = conf/len(result)
				
			data = data.replace(" ", "")
			data = data.replace("\n", "")
			data = data.replace("-", "")

			numbers_count = 0
			letters_count = 0
			for c in data:
				if c.isdigit():
					numbers_count += 1
				elif c.isalpha():
					letters_count += 1
				elif c == '-':
					continue
				else:
					#remove all the characters that are not letters or numbers
					data = data.replace(c, "")

			# if (numbers_count > 2 and numbers_count < 5) and (letters_count > 2 and letters_count < 5):
			if len(data) > 5 and len(data) < 10:
				if conf > CONF_THRESH:
					count += 1
					#print(count,'. valid: ',data)
					plate_options.append(data)
					img[y:y+h, x:x+w] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
					cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

					if(has_numbers(data)):
						plate_found = True
	toc = time.perf_counter()
	times.append(toc-tic)
	#print("Time elapsed: ", toc-tic)
	if not plate_found:
		missed_images.append(filename)


	#print(plate_options)
	
	import re
	regex_normal = re.compile(r'([A-Z]{3}[0-9]{3})')
	regex_new = re.compile(r'([A-Z]{4}[0-9]{3})') 
	regex_all = re.compile(r'//[epvz][\d]{5}$|[a-zA-Z]{3}[\d]{3}$|[a-zA-Z]{4}[\d]{2}$|[a-zA-Z]{5}[\d]{1}$|[mM][\d]{2}[\d]{4}$|(ck|dt|hc|cd|hx|ma|ot|rx|rr|CK|DT|HC|CD|HX|MA|OT|RX|RR)[\d]{2}[\d]{2}$|(c-x|x-a|x-b|x-c|C-X|X-A|X-B|X-C)[\d]{4}$')
	all_detections = idx + 1
	for option in plate_options:
		real_plate = original_filename.split(".")[0].replace("-", "").replace(" ", "").lower()
		#if(real_plate in option.lower()):
		if regex_all.search(option):
			found = regex_all.search(option).group(0)
			print(option + " real: " + real_plate + " regex found: " + found)
			if(found.upper() == real_plate.upper()):
				correct_detections += 1
			break
	
	if(idx % 10 == 0):
		print(idx, " ------ ", "Accuracy: ", correct_detections / all_detections * 100, "%")
		print("Correct detections: ", correct_detections)
		print("All detections: ", all_detections)
	

	'''
	#show img and img_orig side by side concatenated
	img  = imutils.resize(img, height = RESIZE_HEIGHT)
	cv2.imshow('img', img)
	pressed = cv2.waitKey(0)
	
	if pressed == ord('q'):
		break
    '''

# copy missed images to a new folder
#split string by / and take the last element
print("Average time: ", sum(times)/len(times))

''''''
for filename in missed_images:
	shutil.copyfile(filename, os.path.join("data/missed_images/", filename.split('/')[-1]))



#print("Final Accuracy: ", correct_detections / all_detections * 100, "%")

