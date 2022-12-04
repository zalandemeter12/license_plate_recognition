import numpy as np
import cv2
from PIL import Image, ImageEnhance
import paddleocr
import torch
import re

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

class PlateReader:
	RESIZE_HEIGHT = 600
	CONF_THRESH = 0.5


	def __init__(self, device_type='cpu') -> None:		
		if device_type == 'gpu':
			device_type = 'cuda'
		print("Using device: " + device_type)
		self.plate_detector = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best_s_250_final.pt', device=device_type)
		self.plate_detector.eval()

		self.reader = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log = False, max_batch_size = 20, total_process_num = 12, use_mp=True)

	def read(self, img, return_img = False) -> dict:
		plates = self.plate_detector(img, size=self.RESIZE_HEIGHT)
		possible_plates = plates.pandas().xyxy[0]

		found_plates = []
		plate_options = []
		for index, row in possible_plates.iterrows():
			#try text recognition
			x = int(row['xmin'])
			y = int(row['ymin'])
			w = int(row['xmax']) - x
			h = int(row['ymax']) - y

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)		

			roi = gray[y:y+h, x:x+w]

			roi = self.preprocess_plate(roi)			

			data = ""
			conf = 0

			texts = self.reader.ocr(roi, cls=True)
			
			if len(texts) < 1:
				continue
			for line in texts:
				if len(line) < 1:
					continue
				for l in line:					
					data_cand = l[1][0]
					conf_cand = l[1][1]
					data = data + data_cand
					conf += conf_cand
				#print(data)
			conf = conf/len(texts)
				
			data = data.replace(" ", "")
			data = data.replace("\n", "")
			data = data.replace("-", "")

			numbers_count = 0
			letters_count = 0
			for c in data:				
				if not c.isalnum():
					#remove all the characters that are not letters or numbers
					data = data.replace(c, "")

			has_numbers = any(char.isdigit() for char in data)

			if return_img:
				img[y:y+h, x:x+w] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

			if len(data) > 5 and len(data) < 10 and has_numbers and conf > self.CONF_THRESH:
				plate_options.append(data)
								
			
		for option in plate_options:
			found_plate = self.validate_plate(option)
			if found_plate:					
				found_plates.append(found_plate)
		
		if return_img:
			return {'plates': found_plates, 'image': img}
		else:
			return {'plates': found_plates}
			

	def preprocess_plate(self, roi):
		# get roi size
		roi_height, roi_width = roi.shape

		scale = 100 / roi_height
		roi = cv2.resize(roi,dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
		enhancer = ImageEnhance.Contrast(Image.fromarray(roi))
		roi = enhancer.enhance(10)
		roi = np.array(roi)		
		kernel = np.ones((3, 3), np.uint8)
		#erosion
		roi = cv2.erode(roi, kernel)
		#dilation
		kernel = np.ones((3, 3), np.uint8)
		roi = cv2.dilate(roi, kernel, iterations=1)
					
		roi = cv2.resize(roi,(roi_width, roi_height))	

		return roi	

	def validate_plate(self, plate) -> str:
		regex_normal = re.compile(r'([A-Z]{3}[0-9]{3})')
		regex_new = re.compile(r'([A-Z]{4}[0-9]{3})') 
		regex_all = re.compile(r'//[epvz][\d]{5}$|[a-zA-Z]{3}[\d]{3}$|[a-zA-Z]{4}[\d]{2}$|[a-zA-Z]{5}[\d]{1}$|[mM][\d]{2}[\d]{4}$|(ck|dt|hc|cd|hx|ma|ot|rx|rr|CK|DT|HC|CD|HX|MA|OT|RX|RR)[\d]{2}[\d]{2}$|(c-x|x-a|x-b|x-c|C-X|X-A|X-B|X-C)[\d]{4}$')
		
		found = ""

		if regex_all.search(plate):
			found = regex_all.search(plate).group(0)

		return found.upper()
		