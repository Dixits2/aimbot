import cv2
from PIL import ImageGrab
import time
from darkflow.net.build import TFNet
import cv2
import numpy
import pyautogui
import json
import keyboard
import sys


options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 0.5, "gpu": 0.7, "labels": "labels.txt"}

tfnet = TFNet(options)

pyautogui.FAILSAFE = False

is_toggled = True

# loc = (800, 600)
loc = (900, 650)

offset = (100, 100)

f_sum = 0
f_count = 0

while True:
	t1 = time.time()

	if is_toggled:
		grab = ImageGrab.grab()
		img = numpy.array(grab)

		result = tfnet.return_predict(img)

		result = sorted(result, key = lambda i: i['confidence'], reverse=True) 

		result[:] = [d for d in result if d.get('label') == 'person']

		for i in result:
			x1 = i['topleft']['x']
			x2 = i['bottomright']['x']
			y1 = i['topleft']['y']
			y2 = i['bottomright']['y']

			x = x1 + (x2 - x1)/2
			y = y1 + (y2 - y1)/2

			if not ((x >= loc[0] - offset[0] and x <= loc[0] + offset[0]) or (y >= loc[1] - offset[1] and y <= loc[1] + offset[1])):
				# print(str(x) + ', ' + str(y) + ": " + i['label'])

				pyautogui.moveTo(x, y, 0.1)
				break

	t2 = time.time()

	f_sum += (t2 - t1)
	f_count += 1

	# sys.stdout.write('\r' + str(f_count/f_sum))
	# sys.stdout.flush()
