import numpy as np
import cv2

LEFT_PATH = "capture/left/{:06d}.jpg"
RIGHT_PATH = "capture/right/{:06d}.jpg"

CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 720

# TODO: Use more stable identifiers
left = cv2.VideoCapture(0)
##right = cv2.VideoCapture(1)

# Increase the resolution
left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
##right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
##right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
##right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# The distortion in the left and right edges prevents a good calibration, so
# discard the edges
CROP_WIDTH = 960
def cropHorizontal(image):
    return image[:,
            int((1280-CROP_WIDTH)/2):
            int(CROP_WIDTH+(1280-CROP_WIDTH)/2)]

frameId = 0

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    ##if not (left.grab() and right.grab()):
    if not(left.grab()):
        print("No more frames")
        break

    _, leftFrame = left.retrieve()
    #leftFrame = cropHorizontal(leftFrame)
    ##_, rightFrame = right.retrieve()
    ##rightFrame = cropHorizontal(rightFrame)

    height, width = leftFrame.shape[:2]

    lefty = leftFrame[0:height, 0:int(width/2)]
    righty = leftFrame[0:height, int(width/2):width]

    lefty = cropHorizontal(lefty)
    righty = cropHorizontal(righty)

    cv2.imwrite(LEFT_PATH.format(frameId), lefty)
    cv2.imwrite(RIGHT_PATH.format(frameId), righty)

    cv2.imshow('left', lefty)
    cv2.imshow('right', righty)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frameId += 1

left.release()
##right.release()
cv2.destroyAllWindows()
