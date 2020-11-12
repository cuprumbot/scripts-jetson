import sys
import numpy as np
import cv2

REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 2048

if len(sys.argv) != 2:
    print("Syntax: {0} CALIBRATION_FILE".format(sys.argv[0]))
    sys.exit(1)

calibration = np.load(sys.argv[1], allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 720

# TODO: Use more stable identifiers
cam = cv2.VideoCapture(0)

# Increase the resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# The distortion in the left and right edges prevents a good calibration, so
# discard the edges
CROP_WIDTH = 960

startLeft = int(((CAMERA_WIDTH/2) - CROP_WIDTH)/2)
endLeft = startLeft + CROP_WIDTH

startRight = int(startLeft + (CAMERA_WIDTH/2))
endRight = startRight + CROP_WIDTH

print( (startLeft,endLeft,startRight,endRight) )

def cropHorizontal(image):
    return image[:,
            int((1280-CROP_WIDTH)/2):
            int(CROP_WIDTH+(1280-CROP_WIDTH)/2)]

def cropLeft (image):
    return image[:,startLeft:endLeft]

def cropRight (image):
    return image[:,startRight:endRight]

# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images
stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(21)
stereoMatcher.setROI1(leftROI)
stereoMatcher.setROI2(rightROI)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(25)

def updateMD(value):
    print("updateMinDisparity")
    print(value)
    stereoMatcher.setMinDisparity(value)

def updateND(value):
    print("updateNumDisparities")
    print(value)
    stereoMatcher.setNumDisparities(value)

def updateBS(value):
    print("updateBlockSize")
    print(value)
    stereoMatcher.setBlockSize(value)

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    if not cam.grab():
        print("No more frames")
        break

    _, camFrame = cam.retrieve()
    
    #height, origWidth = camFrame.shape[:2]
    #width = int(origWidth/2)

    lefty = cropLeft(camFrame)
    righty = cropRight(camFrame)

    height, width = lefty.shape[:2]
    if (width, height) != imageSize:
        print("Left camera has different size than the calibration data")
        print(width)
        print(height)
        print(imageSize)
        break

    height, width = righty.shape[:2]
    if (width, height) != imageSize:
        print("Right camera has different size than the calibration data")
        print(width)
        print(height)
        print(imageSize)
        break

    fixedLeft = cv2.remap(lefty, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(righty, rightMapX, rightMapY, REMAP_INTERPOLATION)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)

    cv2.imshow('left', fixedLeft)
    #cv2.imshow('right', fixedRight)
    
    #cv2.namedWindow('depth')
    #cv2.createTrackbar('min disp', 'depth', 4, 10, updateMD)
    #cv2.createTrackbar('num disps', 'depth', 128, 200, updateND)
    #cv2.createTrackbar('block size', 'depth', 21, 40, updateBS)

    cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
