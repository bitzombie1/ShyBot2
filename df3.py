from __future__ import print_function
from imutils.video import FPS
import argparse
#import imutils
import cv2
import pyrealsense2 as rs
import numpy as np
from common import clock, draw_str
import serial
import time


# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=1,
	help="Whether or not frames should be displayed")
ap.add_argument("-m", "--mintarg", type=int, default=5,
	help="minimum targets seen before bot reacts")
ap.add_argument("-f", "--showface", type=int, default=0,
	help="Show face targets 0/1")
args = vars(ap.parse_args())
display = args["display"]
minNumTarget = args["mintarg"] 
showFace = args["showface"]

c_width = 320
c_heigth = 240

# open serial connection with Arduino
ser = serial.Serial('/dev/ttyUSB0',9600,timeout=0)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, c_width, c_heigth, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
time.sleep(2)



# get depth device and set depth range
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.motion_range, 99)  #94
motion_range = depth_sensor.get_option(rs.option.motion_range)
print("Motion range: ", motion_range)

depth_sensor.set_option(rs.option.filter_option, 0)
filt = depth_sensor.get_option(rs.option.filter_option)
print("filter option: ", filt)

# get depth scale  do I really need this?
depth_scale = depth_sensor.get_depth_scale()
print ("Depth Scale is: " , depth_scale)

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)


frontFaceRecog = "recogs/haarcascade_frontalface_alt2.xml"
sideFaceRecog = "recogs/haarcascade_profileface.xml"
upperBodRecog = "recogs/haarcascade_upperbody.xml"

target = []  # holds queue of targets(x value,y value, z value, time stamp sec)
expTime = 2  # target expiration time in sec  2
minMotion = 10 # value 0 to 255 minimum ROI movement required to load target
highDepth = 0 # temp var to hold longest depth

# helper functions ***************************
def getFrames():
	try:
		while True:
        		# Wait for a coherent pair of frames: depth and color
			frames = pipeline.wait_for_frames()
			aligned_frames = align.process(frames)
			aligned_depth_frame = aligned_frames.get_depth_frame()
			
			#depth_frame = frames.get_depth_frame()
			color_frame = frames.get_color_frame()
			if not aligned_depth_frame or not color_frame:
				continue
			# Convert images to numpy arrays
			depth_image = np.asanyarray(aligned_depth_frame.get_data())
			color_image = np.asanyarray(color_frame.get_data())
			return (color_image, depth_image)                                            
		
	except Exception as e:
		print("Error grabing frames")
		print(e)
		return ([], [])

def faceDetect(img, cascade):
	rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=0, minSize=(10, 10),#minSize=(10, 10)
                                     flags=cv2.CASCADE_SCALE_IMAGE) #scaleFactor=1.3
	if len(rects) == 0:
		return []
	rects[:,2:] += rects[:,:2]
	return rects

def sendSerial(x,y,z):
	
	arduReady = str(ser.readline())
	if "ok" in arduReady:
		outStr = "g" + str(x) + "/" + str(y) + "/" + str(z)
		ser.write(bytearray(outStr, 'utf-8'))
	#else:
	#	print("return: " + arduReady)

def loadTargets(rectList,depthMat,fgMask):
	for box in rectList:
		if int(cv2.mean(fgMask[box[1]:box[3],box[0]:box[2]])[0]) > minMotion:
			x,y = findCenter(box)
			#z = listMedian(depthMat[y*2])
			z = depthMat[y,x]
			if z == 0:
				z = 16000
			#print(depthMat[y*2])
			time = int(clock())
			target.append([x,y,z,time])
			cv2.circle(fgMask,(x,y),5,(255,0,255))
		
def findHotTarget(targetList):
	if len(targetList) < 3:
		return (0,0,0)
	xLst = []
	yLst = []
	zLst = []
	for targ in targetList:
		xLst.append(targ[0])
		yLst.append(targ[1])
		zLst.append(targ[2])
	x = listMedian(xLst)
	y = listMedian(yLst)
	z = listMedian(zLst)
	#z = targetList[xLst.index(x)][2]
	return (x,y,z)

def killTargets(targetList):
	nowTime = int(clock())
	indx =0
	#print(len(targetList))
	for targ in targetList:
		time = targ[3]
		if (time + expTime) < nowTime:
			targetList.pop(indx)
			indx += 1
		else:
			return
		 
def draw_rects(img, rects, color):
	for x1, y1, x2, y2 in rects:
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def findCenter(box):
	return (int(box[0] + (box[2]-box[0])/2),int(box[1] +(box[3]-box[1])/2))

def flipX(rects):
	rects_out = []
	mid = int(c_width/2)
	for x1, y1, x2, y2 in rects:
		cent = x2 - x1
		if cent > mid:
			x1 = mid -(x1 -mid)
			x2 = mid -(x2 -mid)
		elif cent <= (mid - 1):
			x1 = mid +(mid -x1)
			x2 = mid +(mid -x2)
		rects_out.append([x1,y1,x2,y2]) 
	return rects_out

def listMedian(inList):
	nonZero = []
	for x in inList:
		if x != 0:
			nonZero.append(x)
	cnt = len(nonZero)
	if cnt == 0:
		return cnt
	else:
		return sorted(nonZero)[int(cnt/2)]
		
def findHighDepth(depthMat, hDepth):
	global highDepth 
	for row in depthMat:
		for col in row:
			if col > highDepth:
				highDepth = col
	

if __name__ == '__main__':
	fFCascade = cv2.CascadeClassifier(frontFaceRecog)
	sFCascade = cv2.CascadeClassifier(sideFaceRecog)
	uBCascade = cv2.CascadeClassifier(upperBodRecog)
	fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=16, detectShadows=False )
	
	while True:
		t = clock()
		
		(color_img, depth_img) = getFrames()
		if len(color_img) == 0 or len(depth_img) == 0:
			continue
		
		vis = color_img.copy()
		gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)
		fgmask = fgbg.apply(gray)
				
		rects = faceDetect(gray, fFCascade)
		if showFace == 1:
			draw_rects(vis, rects, (0, 255, 0))
		loadTargets(rects, depth_img, fgmask)
		
		
		rects = faceDetect(gray, sFCascade)
		if showFace == 1:
			draw_rects(vis, rects, (255, 0, 0))
		loadTargets(rects, depth_img, fgmask)
		
		rects = faceDetect(cv2.flip(gray, 1), sFCascade)
		rects = flipX(rects)
		if showFace == 1:
			draw_rects(vis, rects, (0, 0, 255))
		loadTargets(rects, depth_img, fgmask)
		
		rects = faceDetect(gray, uBCascade)
		if showFace == 1:
			draw_rects(vis, rects, (0, 0, 0))
		loadTargets(rects, depth_img, fgmask)
		
		killTargets(target)
		if len(target) > minNumTarget:
			x,y,z = findHotTarget(target)
			cv2.circle(vis,(x,y),5,(255,255,255))
			draw_str(vis, (x+5, y+10), str(z))
			sendSerial(x,y,z)
		
		dt = clock() -t
		draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
		# Show images or not
		if display > 0:
			cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE
			cv2.imshow('RealSense', vis)
			cv2.namedWindow('Motion', cv2.WINDOW_NORMAL)
			cv2.imshow('Motion', fgmask)

		if 0xFF & cv2.waitKey(1) == 27:
			break
	cv2.destroyAllWindows()
	pipeline.stop()



