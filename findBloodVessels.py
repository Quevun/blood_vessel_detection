# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:32:40 2016

@author: queky
"""

import ridgeStrength
import hlpr
import findRidge
import numpy as np
import cv2

scale = np.arange(5,256,1)
lump_size = 1   #odd number
ridge = findRidge(scale)
ridge_str_peak = ridgeStrength(scale,lump_size)

ridge2 = np.zeros(np.shape(ridge_str_peak))
for i in range(np.size(ridge_str_peak,2)):
    ridge2[:,:,i] = ridge[:,:,(lump_size-1)/2+i*lump_size]
    
bin = ridge2*ridge_str_peak

for i in range(np.size(bin,2)):
    cv2.imwrite('output/findBloodVessels_results/vessels'+str(i)+'.jpg',bin[:,:,i]*255)
    
    
    
def findRidge(scale):
    img = cv2.imread('input/IR3/test3.bmp',cv2.IMREAD_GRAYSCALE)
    scaled_img = []
    ridge = np.zeros((np.size(img,0),np.size(img,1),len(scale)))
    
    for i in range(len(scale)):
        scaled_img.append(hlpr.ScaledImage(img,scale[i]))
        ridge[:,:,i] = scaled_img[i].findRidge()
    
    return ridge
    
    
    
def ridgeStrength(scale,lump_size):
    img = cv2.imread('input/IR3/test3.bmp',cv2.IMREAD_GRAYSCALE)
    ridge_str_cuboid = hlpr.RidgeStrCuboid(img,scale)

    ######################################################
    # Scale space derivatives
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
    
def connectRidgePeaks(cuboid):
    for t in range(size(cuboid,2)):
        for