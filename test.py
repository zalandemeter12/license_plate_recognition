import numpy as np
import cv2
import imutils
from PIL import Image, ImageEnhance
import paddleocr

CASCADE_PATH = 'cascade.xml'
RESIZE_HEIGHT = 600
CONF_THRESH = 0.5

# Initiate cascade classifer.
plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)
# Initiate OCR
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log = False, max_batch_size = 20, total_process_num = 12, use_mp=True) # need to run only once to download and load model into memory


# Get image from given path


for z in range(1, 11):
	img = cv2.imread('license_plate_' + str(z) + '.jpg')

	img   = imutils.resize(img, height = RESIZE_HEIGHT)

	img_orig = img.copy()

	# Filter image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



	# Detect plates in img
	plates = plate_cascade.detectMultiScale(image=gray, scaleFactor=1.2, minNeighbors=50, minSize=(1,1), maxSize=(600,600))
	count = 0
	print("Found %d plate candidates:" %len(plates))
	for (x,y,w,h) in plates:

		#try text recognition
		roi = gray[y:y+h, x:x+w]

		enhancer = ImageEnhance.Contrast(Image.fromarray(roi))
		roi = enhancer.enhance(10)

		

		roi = np.array(roi)

		thresh = cv2.threshold(roi, 254, 255, cv2.THRESH_BINARY)[1]
		contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = contours[0] if len(contours) == 2 else contours[1]

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



		data = ""
		conf = 0
		result = ocr.ocr(roi, cls=True)
		if len(result) < 1:
			continue
		for line in result:
			data_cand = line[1][0]
			conf_cand = line[1][1]
			data = data_cand + data
			conf += conf_cand
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
				print(count,'. valid: ',data)
				img[y:y+h, x:x+w] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)



	#show img and img_orig side by side concatenated
	cv2.imshow('img', img)
	cv2.waitKey(0)