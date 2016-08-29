# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:30:49 2016

@author: keisoku
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import anaFunc

img = np.ones((200,200))
img[:,90:110] *= 0
img[:,110:] *= 0.5
smoothed = cv2.GaussianBlur(img,(31,31),10)
#smoothed = 1 - smoothed
cv2.imwrite('img_ridge.jpg',(smoothed*255).astype(np.uint8))
anaFunc.plotImg(smoothed)