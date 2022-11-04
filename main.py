from get_license_plate import get_license_plate
import cv2

res = get_license_plate('high_res_images/TAA-262.jpg')

print(res["plate_options"])
cv2.imshow('img', res["image"])
cv2.waitKey(0)