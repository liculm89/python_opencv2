# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:38:35 2017

@author: mauro
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('ring_cad2.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


template = cv2.imread('uzorak4.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.95
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('hdr_res1.png',img_rgb)

cv2.imshow('detected circles',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()