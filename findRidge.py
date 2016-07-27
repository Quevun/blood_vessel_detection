# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:14:15 2016

@author: keisoku
"""

import cv2
import numpy as np
import hlpr

img = cv2.imread('input/IR3/test3.bmp',cv2.IMREAD_GRAYSCALE)
#img = img[:,100:]
scale = range(5,256,5)
scaled_img = []

for i in range(len(scale)):
    scaled_img.append(hlpr.ScaledImage(img,scale[i]))
    ridge = hlpr.float2uint(scaled_img[i].findRidge())
    cv2.imwrite('output/findRidge_results/ridge' + str(i) + '.jpg',ridge)