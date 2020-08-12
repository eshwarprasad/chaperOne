import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import sys
import RPi.GPIO as GPIO
import time
from yolo_utils import infer_image, show_image

FLAGS = []

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

	parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')


	FLAGS, unparsed = parser.parse_known_args()

	# Download the YOLOv3 models if needed
	if FLAGS.download_model:
		subprocess.call(['./yolov3-coco/get_model.sh'])

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
	# Infer real-time on webcam
	count = 0

	vid = cv.VideoCapture(-1)

	while True:			
		_,frame = vid.read()

		height, width = frame.shape[:2]

		if count == 0:
			frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
	    						height, width, frame, colors, labels, FLAGS)
			count += 1

		else:
			frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
	    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
			count = (count + 1) % 150

		#cv.imshow('webcam', frame)

		classids_cpy = classids

		all_freq = {} 

		for i in classids_cpy: 
			if i in all_freq: 
				all_freq[i] += 1
			else: 
				all_freq[i] = 1


		f=open('coco-labels')
		lines=f.readlines()

		#ULTRASONIC PART GOES HERE

		GPIO.setmode(GPIO.BOARD)

		TRIG = 5
		ECHO = 12

		GPIO.setup(TRIG,GPIO.OUT)
		GPIO.output(TRIG,0)

		GPIO.setup(ECHO,GPIO.IN)

		time.sleep(0.1)


		GPIO.output(TRIG,1)
		time.sleep(0.0001)
		GPIO.output(TRIG,0)

		while GPIO.input(ECHO) == 0:
		    pass
		start = time.time()

		while GPIO.input(ECHO) == 1:
		    pass
		stop = time.time()
		
		if not (stop-start==0):#if only an obstacle is closer, device will alert the user

			for key, value in all_freq.items():
				freq=str(value)
				os.system('espeak "'+freq+' '+lines[key]+'"')#object

		
			steps=((stop - start)*17000)/75; #75 in cm - An average distance a person can cover in one step

			os.system('espeak "'+steps+' steps"')

		GPIO.cleanup()

		
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

		sys.exit(0)
	vid.release()
	#cv.destroyAllWindows()

	

