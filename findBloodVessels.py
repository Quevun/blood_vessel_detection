# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:32:40 2016

@author: queky
"""

import hlpr
import numpy as np
import cv2  
import heapq  
    
def findRidge(scale):
    img = cv2.imread('input/IR3/test3.bmp',cv2.IMREAD_GRAYSCALE)
    scaled_img = []
    ridge = np.zeros((np.size(img,0),np.size(img,1),len(scale)))
    
    for i in range(len(scale)):
        scaled_img.append(hlpr.ScaledImage(img,scale[i]))
        ridge[:,:,i] = scaled_img[i].findRidge()
        cv2.imwrite('output/findRidge_results/exclude)
    
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
    
def nStrongestRidges(n,ridges):
    ridge_str_list = []    
    for ridge in ridges:
        ridge_str_list.append(ridge.getTotalRidgeStr())
    nlargest = heapq.nlargest(n,ridge_str_list)
    strongest = []
    for ridge_str in nlargest:
        index = ridge_str_list.index(ridge_str)
        strongest.append(ridges[index])
    return strongest
        
    
scale = np.arange(20,150,1)
#ridge_cuboid = findRidge(scale)
ridge_str_peak = ridgeStrength(scale)
"""
bin = ridge_cuboid*ridge_str_peak
ridges = connectRidgePeaks(bin)
strongest = nStrongestRidges(50,ridges)

i = 0
for ridge in strongest:
    cv2.imwrite('output/strongest_results/strongest'+str(i)+'.jpg',ridge.getImg())
    i += 1
"""
#for i in range(np.size(bin,2)):
#    cv2.imwrite('output/findBloodVessels_results/vessels'+str(i)+'.jpg',bin[:,:,i]*255)