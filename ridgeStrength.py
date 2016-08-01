# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 13:25:49 2016

@author: keisoku
"""

import cv2
import numpy as np
import hlpr

def ridgeStrength(scale,lump_size):
    img = cv2.imread('input/IR3/test3.bmp',cv2.IMREAD_GRAYSCALE)
    ridge_str_cuboid = hlpr.RidgeStrCuboid(img,scale)

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
    scale_deriv = ridge_str_cuboid.getScaleDeriv()
    scale_deriv2 = ridge_str_cuboid.getScaleDeriv2()
        
    bin1 = np.around(scale_deriv) == 0
    bin2 = scale_deriv2 < 0
    
    ###########################################################################
    #   Lump several adjacent scales together
        
    lump_bin1 = hlpr.BinImgCuboid(bin1).lump(lump_size)
    lump_bin2 = hlpr.BinImgCuboid(bin2).lump(lump_size)
    lump_bin = lump_bin1*lump_bin2
    
    ###########################################################################
        
    #bin1 = bin1.astype(np.uint8) * 255
    #bin2 = bin2.astype(np.uint8) * 255
    #lump_bin = lump_bin.astype(np.uint8)*255

    #for i in range(np.size(lump_bin,2)):
    #    cv2.imwrite('output/ridgeStrength_results/lump_bin'+str(i)+'.jpg',lump_bin[:,:,i])
    ######################################################

    return lump_bin