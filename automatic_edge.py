# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:32:34 2017

@author: mauro_suse
"""

# import the necessary packages
import numpy as np
#import argparse
#import glob
import cv2
 
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged
 
 # construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True,
#	help="path to input dataset of images")
#args = vars(ap.parse_args())
 

 
# loop over the images
#for imagePath in glob.glob(args["images"] + "/*.jpg"):
	# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("ring_cad1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
 
	# apply Canny edge detection using a wide threshold, tight
	# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)
 
	# show the images
#cv2.imshow("Original", image)
cv2.imshow("Edges", np.hstack([wide, tight, auto]))

circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(auto,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(auto,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',auto)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.waitKey(0)