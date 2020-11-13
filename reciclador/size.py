# import the necessary packages
from __future__ import print_function
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import time

refWidth = 1.0
blurSize = 5
contourAreaThreshold = 200


# CALIBRATION
REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 2048

calibration = np.load("calibration.npz", allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

print(imageSize)
# CALIBRATION



# ls /dev | grep video
cam = cv2.VideoCapture(0)

# resolucion camara dual
CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 720
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# descartar bordes
CROP_WIDTH = 960
def cropHorizontal(image):
	return image[:,
			int((1280-CROP_WIDTH)/2):
			int(CROP_WIDTH+(1280-CROP_WIDTH)/2)]

def cropAfterCorrection(image):
	return image[60:,:900]

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



while(True):
	# terminar si la camara falla y no se puede capturar imagen
	if not(cam.grab()):
		print("No more frames")
		break

	# obtener imagen
	_, camFrame = cam.retrieve()

	height, width = camFrame.shape[:2]

	lefty = camFrame[0:height, 0:int(width/2)]
	#righty = camFrame[0:height, int(width/2):width]

	image = cropHorizontal(lefty)
	#cv2.imshow('Image', image)

	image = cv2.remap(image, leftMapX, leftMapY, REMAP_INTERPOLATION)
	image = cropAfterCorrection(image)

	
	# load our input image, convert it to grayscale, and blur it slightly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (blurSize, blurSize), 0)
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	cv2.imshow("Edges", edged)


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
	# cada iteracion se limpia el tamano referencia
	pixelsPerMetric = None
	#colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))


	#orig = image.copy()
	orig = image
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
		
		# show the original coordinates
		#print("Object #{}:".format(i + 1))
		#print(box)

		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, color, 2)

		# loop over the original points and draw them
		#for (x, y) in box:
		#	cv2.circle(orig, (int(x), int(y)), 2, (0, 0, 255), -1)
		# draw the object num at the top-left corner
		cv2.putText(image, "Object #{}".format(numBoxes),
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
		# draw the midpoints on the image
		#cv2.circle(orig, (int(tltrX), int(tltrY)), 2, (0, 0, 255), -1)
		#cv2.circle(orig, (int(blbrX), int(blbrY)), 2, (0, 0, 255), -1)
		#cv2.circle(orig, (int(tlblX), int(tlblY)), 2, (0, 0, 255), -1)
		#cv2.circle(orig, (int(trbrX), int(trbrY)), 2, (0, 0, 255), -1)

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
		cv2.imshow("Image", orig)
		#cv2.waitKey(0)

	if cv2.waitKey(50) & 0xFF == ord('q'):
		break

	if cv2.waitKey(50) & 0xFF == ord('z'):
		time.sleep(5)

cam.release()
cv2.destroyAllWindows()