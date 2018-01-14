from __future__ import print_function
from imutils.video import FPS
import argparse
import imutils
import cv2
import pyrealsense2 as rs
import numpy as np
from common import clock, draw_str
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
ap.add_argument("-t", "--thread", type=int, default=-1,
	help="Whether or not threads are checked")
args = vars(ap.parse_args())

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
time.sleep(2)

frontFaceRecog = "recogs/haarcascade_frontalface_alt2.xml"
sideFaceRecog = "recogs/haarcascade_profileface.xml"
upperBodRecog = "recogs/haarcascade_upperbody.xml"

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def getFrames():
	try:
		while True:
        		# Wait for a coherent pair of frames: depth and color
			frames = pipeline.wait_for_frames()
			depth_frame = frames.get_depth_frame()
			color_frame = frames.get_color_frame()
			if not depth_frame or not color_frame:
				continue
			# Convert images to numpy arrays
			depth_image = np.asanyarray(depth_frame.get_data())
			color_image = np.asanyarray(color_frame.get_data())
			return (color_image, depth_image)
	
	
	except Exception as e:
		print("Error grabing frames")
		print(e)
		return ([], [])

def faceDetect(img, cascade):
	rects = cascade.detectMultiScale(img, scaleFactor=1.4, minNeighbors=1, minSize=(10, 10),#minSize=(10, 10)
                                     flags=cv2.CASCADE_SCALE_IMAGE) #scaleFactor=1.3
	if len(rects) == 0:
		return []
	rects[:,2:] += rects[:,:2]
	return rects

def draw_rects(img, rects, color):
	for x1, y1, x2, y2 in rects:
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def flipX(rects):
	rects_out = []
	for x1, y1, x2, y2 in rects:
		cent = x2 - x1
		if cent > 200:
			x1 = 200 -(x1 -200)
			x2 = 200 -(x2 -200)
		elif cent <= 199:
			x1 = 200 +(200 -x1)
			x2 = 200 +(200 -x2)
		rects_out.append([x1,y1,x2,y2]) 
	return rects_out

if __name__ == '__main__':
	fFCascade = cv2.CascadeClassifier(frontFaceRecog)
	sFCascade = cv2.CascadeClassifier(sideFaceRecog)
	uBCascade = cv2.CascadeClassifier(upperBodRecog)
	while True:
		t = clock()
		
		(color_img, depth_img) = getFrames()
		if len(color_img) == 0 or len(depth_img) == 0:
			continue
		color_img_small = imutils.resize(color_img, width=min(400, color_img.shape[1]))
		
		vis = color_img_small.copy()
		gray = cv2.cvtColor(color_img_small, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)
				
		

		rects = faceDetect(gray, fFCascade)
		draw_rects(vis, rects, (0, 255, 0))

		
		rects = faceDetect(gray, sFCascade)
		draw_rects(vis, rects, (255, 0, 0))

		
		rects = faceDetect(cv2.flip(gray, 1), sFCascade)
		rects = flipX(rects)
		draw_rects(vis, rects, (0, 0, 255))
		
		
		rects = faceDetect(gray, uBCascade)
		draw_rects(vis, rects, (0, 0, 0))

		# detect peds in the image
		#(rects, weights) = hog.detectMultiScale(gray, winStride=(8, 8),
		#padding=(16, 16), scale=1.05)
		#for (x, y, w, h) in rects:
			#cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

		dt = clock() -t
		draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
		# Show images
		cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE
		cv2.imshow('RealSense', vis)
		if 0xFF & cv2.waitKey(1) == 27:
			break
	cv2.destroyAllWindows()
	pipeline.stop()



