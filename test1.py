# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:00:12 2017

@author: mauro
"""

import numpy as np
import cv2
import matplotlib
print(matplotlib.matplotlib_fname())
#matplotlib.use("Agg") 
from matplotlib import pyplot as plt
cv2.ocl.setUseOpenCL(False)

img1 = cv2.imread('ring_cad2.jpg', 0)
img2 = cv2.imread('uzorak1.jpg', 0)


#gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#blurred1 = cv2.GaussianBlur(gray1, (3, 3), 0)

detector = cv2.AKAZE_create()
#detector = cv2.ORB_create()


kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)


draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matches,
                   flags = 0)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:20], None, flags=2)

plt.imshow(img3),plt.show()