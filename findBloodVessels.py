# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:32:40 2016

@author: queky
"""

import hlpr
import numpy as np
import cv2    
    
def findRidge(scale):
    img = cv2.imread('input/IR3/test3.bmp',cv2.IMREAD_GRAYSCALE)
    scaled_img = []
    ridge = np.zeros((np.size(img,0),np.size(img,1),len(scale)))
    
    for i in range(len(scale)):
        scaled_img.append(hlpr.ScaledImage(img,scale[i]))
        ridge[:,:,i] = scaled_img[i].findRidge()
    
    return ridge
    
    
    
def ridgeStrength(scale):
    img = cv2.imread('input/IR3/test3.bmp',cv2.IMREAD_GRAYSCALE)
    ridge_str_cuboid = hlpr.RidgeStrCuboid(img,scale)

    ######################################################
    # Scale space derivatives
    scale_deriv = ridge_str_cuboid.getScaleDeriv()
    scale_deriv2 = ridge_str_cuboid.getScaleDeriv2()
        
    bin1 = np.around(scale_deriv) == 0
    bin2 = scale_deriv2 < 0

    ######################################################

    return (bin1*bin2)*ridge_str_cuboid.cuboid
    
def connectRidgePeaks(cuboid):
    ridges = []
    hlpr.Ridge.setCuboid(cuboid)
    it = np.nditer(cuboid, flags=['multi_index'])
    while not it.finished:
        if hlpr.Ridge.getCuboid()[it.multi_index] > 0:
            pixel = hlpr.Pixel(it.multi_index,it[0])
            ridges.append(hlpr.Ridge(pixel))
            ridges[-1].growRidge()
        it.iternext()
        
    return ridges
    
scale = np.arange(50,100,1)
ridge = findRidge(scale)
ridge_str_peak = ridgeStrength(scale)
bin = ridge*ridge_str_peak
ridges = connectRidgePeaks(bin)
#for i in range(np.size(bin,2)):
#    cv2.imwrite('output/findBloodVessels_results/vessels'+str(i)+'.jpg',bin[:,:,i]*255)