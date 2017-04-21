# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:49:44 2017

@author: mauro
"""

import cv2 
import numpy as np

img = cv2.imread("ring_cad2.jpg", 0)

ret,thresh = cv2.threshold(img,127,255,0)
contours = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
print(M)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

#area = cv2.contourArea(cnt)

#perimeter = cv2.arcLength(cnt,True)

(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(img,center,radius,(0,255,0),2)