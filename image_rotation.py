# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:45:32 2017
@author: mauro
"""

# import the necessary packages
import numpy as np
#import argparse
import imutils
import cv2
import statistics

image = cv2.imread("ring_cad2.jpg")

counter = 0
font = cv2.FONT_HERSHEY_SIMPLEX
kutevi = []

template = cv2.imread('uzorak4.jpg',0)
w, h = template.shape[::-1]
 
# loop over the rotation angles
for angle in np.arange(0, 361, 1):
    rotated = imutils.rotate(image, angle)
    #rotated = imutils.rotate_bound(image, angle) # ISPRAVNA ROTACIJA
    img_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.95
    
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        print("Found")
        counter += 1
        kutevi.append(angle)        
        cv2.rectangle(rotated, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    print(angle)
    string = "Kut rotacije: " + str(angle)
   
    cv2.putText(rotated,string,(200,100), font, 1,(0,255,255),2)



    cv2.imshow("Rotating...", rotated)
    cv2.waitKey(0)
 
clean_kutevi = list(set(kutevi)) 
med_kut = statistics.median(clean_kutevi)

print("Median kuta : " + str(med_kut))
print("Ukupno pronađenih patterna : " + str(counter))

"""
for angle in np.arange(0, 360, 1):
	rotated = imutils.rotate_bound(image, angle)
	cv2.imshow("Rotated (Correct)", rotated)
	cv2.waitKey(0)
"""