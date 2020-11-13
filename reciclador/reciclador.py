# size map
from __future__ import print_function
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
import numpy as np
import imutils
import time

# general
import sys
import argparse
import subprocess
import cv2

# clasificacion
import jetson.inference
import jetson.utils



# size map
refWidth = 1.0
blurSize = 5
contourAreaThreshold = 200

# general
WINDOW_NAME = 'Reciclador'
CANNY_NAME = 'Edges'
SIZE_NAME = 'Coords'

# clasificacion
net = jetson.inference.imageNet('resnet-18')
camera = jetson.utils.videoSource("csi://0")
display = jetson.utils.videoOutput("display://0")

# punto medio
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# ordenar puntos
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

def parse_args():
	# Parse input arguments
	desc = 'Capture and display live camera video on Jetson TX2/TX1'
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--usb', dest='use_usb',
						help='use USB webcam (remember to also set --vid)',
						action='store_true')
	parser.add_argument('--vid', dest='video_dev',
						help='device # of USB webcam (/dev/video?) [1]',
						default=0, type=int)
	parser.add_argument('--width', dest='image_width',
						help='image width [1920]',
						default=1920, type=int)
	parser.add_argument('--height', dest='image_height',
						help='image height [1080]',
						default=1080, type=int)
	args = parser.parse_args()
	return args

def open_cam_usb(dev, width, height):
	# We want to set width and height here, otherwise we could just do:
	#     return cv2.VideoCapture(dev)
	gst_str = ('v4l2src device=/dev/video{} ! '
			   'video/x-raw, width=(int){}, height=(int){} ! '
			   'videoconvert ! appsink').format(dev, width, height)
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_onboard(width, height):
	gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
	if 'nvcamerasrc' in gst_elements:
		# On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
		gst_str = ('nvcamerasrc ! '
				   'video/x-raw(memory:NVMM), '
				   'width=(int)2592, height=(int)1458, '
				   'format=(string)I420, framerate=(fraction)30/1 ! '
				   'nvvidconv ! '
				   'video/x-raw, width=(int){}, height=(int){}, '
				   'format=(string)BGRx ! '
				   'videoconvert ! appsink').format(width, height)
	elif 'nvarguscamerasrc' in gst_elements:
		gst_str = ('nvarguscamerasrc ! '
				   'video/x-raw(memory:NVMM), '
				   'width=(int)1920, height=(int)1080, '
				   'format=(string)NV12, framerate=(fraction)30/1 ! '
				   'nvvidconv flip-method=2 ! '
				   'video/x-raw, width=(int){}, height=(int){}, '
				   'format=(string)BGRx ! '
				   'videoconvert ! appsink').format(width, height)
	else:
		raise RuntimeError('onboard camera source not found!')
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_window(width, height):
	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
	cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)
	
	cv2.namedWindow(CANNY_NAME, cv2.WINDOW_NORMAL)
	cv2.setWindowTitle(CANNY_NAME, CANNY_NAME)
	
	cv2.namedWindow(SIZE_NAME, cv2.WINDOW_NORMAL)
	cv2.setWindowTitle(SIZE_NAME, SIZE_NAME)
	
	cv2.resizeWindow(WINDOW_NAME, 640, 360)
	cv2.moveWindow(WINDOW_NAME, 0, 0)
	
	cv2.resizeWindow(CANNY_NAME, 640, 360)
	cv2.moveWindow(CANNY_NAME, 640, 0)
	
	cv2.resizeWindow(SIZE_NAME, 640, 360)
	cv2.moveWindow(SIZE_NAME, 640, 360)


def read_cam(cap):
	show_help = False
	full_scrn = False
	help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
	font = cv2.FONT_HERSHEY_PLAIN
	while True:
		if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
			# Check to see if the user has closed the window
			# If yes, terminate the program
			break
		
		_, img = cap.read() # grab the next image frame from camera
		
		#if show_help:
		#	cv2.putText(img, help_text, (11, 20), font,
		#				1.0, (32, 32, 32), 4, cv2.LINE_AA)
		#	cv2.putText(img, help_text, (10, 20), font,
		#				1.0, (240, 240, 240), 1, cv2.LINE_AA)
		
		img2 = jetson.utils.cudaFromNumpy(img)
		class_idx, confidence = net.Classify(img2)
		class_desc = net.GetClassDesc(class_idx)
		print('        clase detectada        ', class_desc)
		
####### DETECT EDGES
		# load our input image, convert it to grayscale, and blur it slightly
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (blurSize, blurSize), 0)
		# perform edge detection, then perform a dilation + erosion to
		# close gaps in between object edges
		edged = cv2.Canny(gray, 50, 100)
		edged = cv2.dilate(edged, None, iterations=1)
		edged = cv2.erode(edged, None, iterations=1)
		cv2.imshow(CANNY_NAME, edged)

		# find contours in the edge map
		# edged.copy()
		cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		# sort the contours from left-to-right and initialize the bounding box
		# point colors
		try:
			(cnts, _) = contours.sort_contours(cnts)
		except:
			# no se encontraron contornos
			# TO DO: integracion
			print('no hay contornos')
			continue
####### EDGES DETECTED!

####### GET SIZES
		# cada iteracion se limpia el tamano referencia
		pixelsPerMetric = None

		orig = img.copy()
		numBoxes = 0
		# loop over the contours individually
		for (i, c) in enumerate(cnts):
			# if the contour is not sufficiently large, ignore it
			if cv2.contourArea(c) < contourAreaThreshold:
				continue

			color = (0, 255, 0)
			if numBoxes == 0:
				color = (0, 0, 255)
			elif numBoxes > 3:
				break
			numBoxes = numBoxes + 1

			# compute the rotated bounding box of the contour, then
			# draw the contours
			box = cv2.minAreaRect(c)
			box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
			box = np.array(box, dtype="int")

			box = perspective.order_points(box)
			cv2.drawContours(orig, [box.astype("int")], -1, color, 2)

			# draw the object num at the top-left corner
			cv2.putText(orig, "Object #{}".format(numBoxes),
				(int(box[0][0] - 15), int(box[0][1] - 15)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
			
			# unpack the ordered bounding box
			(tl, tr, br, bl) = box
			# compute the midpoint between the top-left and top-right points,
			# followed by the midpoint between bottom-left and bottom-right
			(tltrX, tltrY) = midpoint(tl, tr)
			(blbrX, blbrY) = midpoint(bl, br)
			# compute the midpoint between the top-left and top-right points,
			# followed by the midpoint between the top-righ and bottom-right
			(tlblX, tlblY) = midpoint(tl, bl)
			(trbrX, trbrY) = midpoint(tr, br)

			# compute the Euclidean distance between the midpoints
			dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
			dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

			(centerX, centerY) = (0, 0)

			if dA < dB:
				cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
				(centerX, centerY) = midpoint((tltrX,tltrY), (blbrX,blbrY))
			else:
				cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
				(centerX, centerY) = midpoint((tlblX, tlblY), (trbrX,trbrY))

			cv2.circle(orig, (int(centerX), int(centerY)), 2, (0, 0, 255), 5)
			
			# inicializar medida, la referencia es la base del primer objeto
			# refWidth deberia ser el tamano en cm del objeto referencia
			if pixelsPerMetric is None:
				pixelsPerMetric = dB / refWidth

			# compute the size of the object
			dimA = dA / pixelsPerMetric
			dimB = dB / pixelsPerMetric
			# draw the object sizes on the image
			cv2.putText(orig, "{:.1f}".format(dimA),
				(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)
			cv2.putText(orig, "{:.1f}".format(dimB),
				(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)
			cv2.putText(orig, "{x},{y}".format(x = int(centerX), y = int(centerY)),
				(int(centerX), int(centerY)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)

			# show the image
			cv2.imshow(SIZE_NAME, orig)

		#if cv2.waitKey(50) & 0xFF == ord('q'):
		#	break

		#if cv2.waitKey(50) & 0xFF == ord('z'):
		#	time.sleep(5)		
####### GOT THE SIZES!

		cv2.putText(img, '{:s} {:f}'.format(class_desc, confidence*100), (11, 50), font,
						3.0, (32, 32, 32), 4, cv2.LINE_AA)
		cv2.putText(img, '{:s} {:f}'.format(class_desc, confidence*100), (10, 50), font,
						3.0, (240, 240, 240), 1, cv2.LINE_AA)
		cv2.imshow(WINDOW_NAME, img)
		
		

		
		
		
		key = cv2.waitKey(10)
		if key == 27: # ESC key: quit program
			break
		elif key == ord('H') or key == ord('h'): # toggle help message
			show_help = not show_help
		elif key == ord('F') or key == ord('f'): # toggle fullscreen
			full_scrn = not full_scrn
			if full_scrn:
				cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
									  cv2.WINDOW_FULLSCREEN)
			else:
				cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
									  cv2.WINDOW_NORMAL)










def main():
	args = parse_args()
	print('Called with args:')
	print(args)
	print('OpenCV version: {}'.format(cv2.__version__))

	if args.use_usb:
		cap = open_cam_usb(args.video_dev,
						   args.image_width,
						   args.image_height)
	else: # by default, use the Jetson onboard camera
		cap = open_cam_onboard(args.image_width,
							   args.image_height)

	if not cap.isOpened():
		sys.exit('Failed to open camera!')

	open_window(args.image_width, args.image_height)
	read_cam(cap)

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
