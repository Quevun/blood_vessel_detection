# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 13:25:49 2016

@author: keisoku
"""

import cv2
import numpy as np
import hlpr

def ridgeStrength():
    img = cv2.imread('input/IR3/test3.bmp',cv2.IMREAD_GRAYSCALE)
    #img = img[:,100:]
    scale = np.arange(5,256,5)
    scaled_img = []
    ridge_strength = np.zeros((np.size(img,0),np.size(img,1),len(scale)))
    
    for i in range(len(scale)):
        scaled_img.append(hlpr.ScaledImage(img,scale[i]))
        #cv2.imwrite('output/ridgeStrength_results/ridgeStrength' + str(i) + '.jpg',hlpr.float2uint(scaled_img[i].getRidgeStrength()))
        ridge_strength[:,:,i] = scaled_img[i].getRidgeStrength()

    ######################################################
    # Scale space derivatives
    """#Using Sobel
    scale_deriv = np.zeros(np.shape(ridge_strength))
    scale_deriv2 = np.zeros(np.shape(ridge_strength))
    bin1 = np.zeros(np.shape(ridge_strength))
    bin2 = np.zeros(np.shape(ridge_strength))
    bin3 = np.zeros(np.shape(ridge_strength))
    for i in range(np.size(ridge_strength,1)):
        slice = ridge_strength[:,i,:]
        slice = cv2.GaussianBlur(slice,(19,19),3)
        scale_deriv[:,i,:] = cv2.Sobel(slice,cv2.CV_64F,1,0)
        scale_deriv2[:,i,:] = cv2.Sobel(slice,cv2.CV_64F,2,0)                
        #bin1[:,i,:] = np.around(scale_deriv[:,i,:],2) == 0
        bin2[:,i,:] = scale_deriv2[:,i,:] < 0
        #bin3[:,i,:] = np.logical_and(bin1,bin2)
        
    #bin3 = bin3.astype(np.uint8) * 255
    #bin1 = bin1.astype(np.uint8) * 255
    bin2 = bin2.astype(np.uint8) * 255
    
    for i in range(len(scale)):
        #cv2.imwrite('output/ridgeStrength_results/bin_one'+str(i)+'.jpg',bin1[:,:,i])
        cv2.imwrite('output/ridgeStrength_results/bin_two'+str(i)+'.jpg',bin2[:,:,i])
    """

    #Subtract scale images directly
    scale_deriv = np.zeros(np.shape(ridge_strength))
    scale_deriv2 = np.zeros(np.shape(ridge_strength))
    
    max_i = np.size(ridge_strength,2)-1
    for i in range(np.size(ridge_strength,2)):
        scale_deriv[:,:,i] = ridge_strength[:,:,(i+1)%max_i]-ridge_strength[:,:,i-1]
        
    for i in range(np.size(ridge_strength,2)):    
        scale_deriv2[:,:,i] = scale_deriv[:,:,(i+1)%max_i]-scale_deriv[:,:,i-1]
        
    bin1 = np.around(scale_deriv) == 0
    bin2 = scale_deriv2 < -0.3
    
    ###########################################################################
    #   Lump several adjacent scales together
    
    lump_size = 3
    for i in range(0,np.size(bin1,2)-lump_size+1,lump_size):
        #temp1 = np.logical_or(bin1[:,:,i],bin1[:,:,i+1])
        #temp2 = np.logical_or(temp1,bin1[:,:,i+2])
        #temp3 = np.logical_or(temp2,bin1[:,:,i+3])
        #temp4 = np.logical_or(temp3,bin1[:,:,i+4])
        #temp4 = temp4.astype(np.uint8)*255
        lump_bin1 = bin1[:,:,i]
        lump_bin2 = bin2[:,:,i]
        for j in range(1,lump_size):
            lump_bin1 += bin1[:,:,i+j]
            lump_bin2 += bin2[:,:,i+j]
        lump_bin1 = lump_bin1 > 0   # Equivalent to 'or' operation for one whole lump
        lump_bin2 = lump_bin2 > 0
        lump_bin = np.logical_and(lump_bin1,lump_bin2).astype(np.uint8)*255
        cv2.imwrite('output/ridgeStrength_results/lump_bin'+str(i/lump_size)+'.jpg',lump_bin)

    ###########################################################################
        
    bin1 = bin1.astype(np.uint8) * 255
    bin2 = bin2.astype(np.uint8) * 255
        
    #for i in range(np.size(bin2,2)):
    #    cv2.imwrite('output/ridgeStrength_results/bin_one'+str(i)+'.jpg',bin1[:,:,i])
    ######################################################
    
    return ridge_strength