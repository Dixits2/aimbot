# from darkflow.net.build import TFNet
# import cv2

# options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 0.1}

# tfnet = TFNet(options)

# imgcv = cv2.imread("./sample_img/sample_dog.jpg")
# result = tfnet.return_predict(imgcv)
# print(result)



import cv2
from PIL import ImageGrab
import time
from darkflow.net.build import TFNet
import cv2
import numpy
import pyautogui
import json
import keyboard

# options = {"model": "cfg/tiny-yolo.cfg", "load": "bin/yolo-tiny.weights", "threshold": 0.1}
options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 0.3, "gpu": 0.4, "labels": "labels.txt"}

tfnet = TFNet(options)

pyautogui.FAILSAFE = False

is_toggled = True

while True:

	if keyboard.is_pressed('n'):
		is_toggled = not is_toggled
		print("toggled is " + str(is_toggled))

	if is_toggled:
		grab = ImageGrab.grab()
		img = numpy.array(grab)

		result = tfnet.return_predict(img)

		highest_index = -1
		highest_index_val = -1

		if keyboard.is_pressed('n'):
			is_toggled = not is_toggled
			print("toggled is " + str(is_toggled))

		for i in range(len(result)):
			if result[i]['label'] == 'person':
				if highest_index == -1:
					highest_index_val = result[i]['confidence']
					highest_index = i
				else: 
					if highest_index_val < result[i]['confidence']:
						highest_index_val = result[i]['confidence']
						highest_index = i

		if highest_index != -1:
			i = result[highest_index]

			x1 = i['topleft']['x']
			x2 = i['bottomright']['x']
			y1 = i['topleft']['y']
			y2 = i['bottomright']['y']

			x = x1 + (x2 - x1)/2
			y = y1 + (y2 - y1)/2

			pyautogui.moveTo(x, y)