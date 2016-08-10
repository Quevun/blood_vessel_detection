# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:49:37 2016

@author: keisoku
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:32:40 2016

@author: queky
"""

import hlpr
import numpy as np
import cv2  
import heapq  
import anaFunc
import test
    
def findRidge(scale,img):
    scaled_img = []
    ridge = np.zeros((np.size(img,0),np.size(img,1),len(scale)))
    
    for i in range(len(scale)):
        scaled_img.append(hlpr.ScaledImage(img,scale[i]))
        ridge[:,:,i] = scaled_img[i].findRidge()
        cv2.imwrite('output/findRidge_results/marker'+str(i)+'.jpg',ridge[:,:,i].astype(np.uint8)*255)
    return ridge
    
def ridgeStrength(scale,img):
    ridge_str_cuboid = hlpr.RidgeStrCuboid(img,scale)

    ######################################################
    # Scale space derivatives
    scale_deriv = ridge_str_cuboid.getScaleDeriv()
    scale_deriv2 = ridge_str_cuboid.getScaleDeriv2()
        
    #bin1 = np.around(scale_deriv) == 0
    scale_deriv = test.normAxis2(scale_deriv)
    bin1 = abs(scale_deriv) < 0.01
    #bin1 = np.ones(np.shape(ridge_str_cuboid))  # testing purpose
    bin2 = scale_deriv2 < 0
    bin3 = (bin1*bin2)*ridge_str_cuboid.cuboid
    ######################################################
    bin1 = bin1.astype(np.uint8)*255
    bin2 = bin2.astype(np.uint8)*255
    bin4 = (bin3 > 0).astype(np.uint8)*255
    #anaFunc.plotRidgeStrAlongScale(scale_deriv[:,:,2:-4],[(334,230),(293,291),(511,254),(394,350)])
    #ridge_str_cuboid = (ridge_str_cuboid.cuboid/np.amax(ridge_str_cuboid.cuboid)*255).astype(np.uint8)
    for i in range(len(scale)):
        #cv2.imwrite('output/ridgeStrength_results/bin_one'+str(i)+'.jpg',bin1[:,:,i])
        #cv2.imwrite('output/ridgeStrength_results/bin_two'+str(i)+'.jpg',bin2[:,:,i])
        cv2.imwrite('output/ridgeStrength_results/marker'+str(i)+'.jpg',bin4[:,:,i])
    return bin3
    
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
        
img = cv2.imread('input/marker.bmp',cv2.IMREAD_GRAYSCALE)
#img = cv2.pyrDown(img)

scale = np.arange(1,100,1)
ridge_cuboid = findRidge(scale,img)
#ridge_str_peak = ridgeStrength(scale,img)

#bin = ridge_cuboid*ridge_str_peak
#for i in range(np.size(bin,2)):
#    cv2.imwrite('output/findBloodVessels_results/marker'+str(i)+'.jpg',bin[:,:,i]*255)
"""
ridges = connectRidgePeaks(bin)
strongest = nStrongestRidges(100,ridges)
#i = 0
#for ridge in strongest:
#    cv2.imwrite('output/strongest_results/strongest'+str(i)+'.jpg',ridge.getImg())
#    i += 1

combined = np.zeros(np.shape(img))
for ridge in strongest:
    combined += ridge.getImg()
combined = combined > 0
combined = combined.astype(np.uint8)*255
cv2.imwrite('combined.jpg',combined)
"""