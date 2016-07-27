# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 13:25:49 2016

@author: keisoku
"""

import cv2
import numpy as np
import hlpr

img = cv2.imread('input/IR3/test3.bmp',cv2.IMREAD_GRAYSCALE)
#img = img[:,100:]
scale = np.arange(5,256,0.1)
scaled_img = []
ridge_strength = np.zeros((np.size(img,0),np.size(img,1),len(scale)))

for i in range(len(scale)):
    scaled_img.append(hlpr.ScaledImage(img,scale[i]))
    #cv2.imwrite('output/ridgeStrength_results/ridgeStrength' + str(i) + '.jpg',hlpr.float2uint(scaled_img[i].getRidgeStrength()))
    ridge_strength[:,:,i] = scaled_img[i].getRidgeStrength()
    
scale_deriv = np.zeros(np.shape(ridge_strength))
scale_deriv2 = np.zeros(np.shape(ridge_strength))
bin1 = np.zeros(np.shape(ridge_strength))
bin3 = np.zeros(np.shape(ridge_strength))
    
for i in range(np.size(ridge_strength,1)):
    slice = ridge_strength[:,i,:]
    scale_deriv[:,i,:] = cv2.Sobel(slice,cv2.CV_64F,1,0)
    scale_deriv2[:,i,:] = cv2.Sobel(slice,cv2.CV_64F,2,0)
    bin1[:,i,:] = hlpr.float2uint(scale_deriv[:,i,:]) == 0
    #bin2 = scale_deriv2[:,i,:] <= 0
    #bin3[:,i,:] = np.logical_and(bin1,bin2)
    
#bin3 = bin3.astype(np.uint8) * 255
bin1 = bin1.astype(np.uint8) * 255

for i in range(len(scale)):
    cv2.imwrite('output/ridgeStrength_results/scale_deriv_zero'+str(i)+'.jpg',bin1[:,:,i])