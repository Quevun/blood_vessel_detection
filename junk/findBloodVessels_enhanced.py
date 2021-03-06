# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:10:38 2016

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
        ridge[:,:,i] = scaled_img[i].findRidge('curvature')
        cv2.imwrite('output/findRidge_results/arm_enhanced'+str(i)+'.jpg',ridge[:,:,i].astype(np.uint8)*255)
    return ridge
    
def ridgeStrength(scale,img):
    ridge_str_cuboid = hlpr.RidgeStrCuboid(img,scale)

    ######################################################
    # Scale space derivatives
    scale_deriv = ridge_str_cuboid.getScaleDeriv()
    scale_deriv2 = ridge_str_cuboid.getScaleDeriv2()
    
    #bin1 = np.around(scale_deriv) == 0
    bin1 = hlpr.scaleDerivZero(scale_deriv)
    #bin1 = np.ones(np.shape(ridge_str_cuboid))  # testing purpose
    bin2 = scale_deriv2 < 0
    bin3 = (bin1*bin2[:,:,:-1])*ridge_str_cuboid.cuboid[:,:,:-1]
    ######################################################
    #bin1 = bin1.astype(np.uint8)*255
    #bin2 = bin2.astype(np.uint8)*255
    bin4 = (bin3 > 0).astype(np.uint8)*255
    #anaFunc.plotRidgeStrAlongScale(scale_deriv[:,:,2:-4],[(334,230),(293,291),(511,254),(394,350)])
    
    #coords = anaFunc.getNeighbourCoords((481,223),5)    #(481, 223),(632, 333),(415, 88)
    #anaFunc.plotRidgeStrAlongScale(scale_deriv[:,:,1:-1],coords)
    
    #anaFunc.plotRidgeStrAlongScale(ridge_str_cuboid.cuboid,coords)
    #ridge_str_cuboid = (ridge_str_cuboid.cuboid/np.amax(ridge_str_cuboid.cuboid)*255).astype(np.uint8)
    #for i in range(len(scale)-1):
    #    cv2.imwrite('output/ridgeStrength_results/bin_one'+str(i)+'.jpg',bin1[:,:,i])
    for i in range(len(scale)-1):
        #cv2.imwrite('output/ridgeStrength_results/bin_two'+str(i)+'.jpg',bin2[:,:,i])
        cv2.imwrite('output/ridgeStrength_results/arm_enhanced'+str(i)+'.jpg',bin4[:,:,i])
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
        
img = cv2.imread('input/IR3/test3.bmp',cv2.IMREAD_GRAYSCALE)
img = test.enhanceRidges(img[:,300:].astype(np.float64))

scale = np.arange(1,250,5)
ridge_cuboid = findRidge(scale,img)
ridge_str_peak = ridgeStrength(scale,img)

#bin = ridge_cuboid[:,:,:-1]*ridge_str_peak**(0.25)
#bin2 = (bin > 0).astype(np.uint8)*255
#for i in range(np.size(bin2,2)):
#    cv2.imwrite('output/findBloodVessels_results/arm_big_interval'+str(i)+'.jpg',bin2[:,:,i])
"""
ridges = connectRidgePeaks(bin[:,:,2:-2])
strongest = nStrongestRidges(100,ridges)
i = 0
for ridge in strongest:
    cv2.imwrite('output/strongest_results/arm_enhanced'+str(i)+'.jpg',ridge.getImg())
    i += 1

combined = np.zeros(np.shape(img))
for ridge in strongest:
    combined += ridge.getImg()
combined = combined > 0
combined = combined.astype(np.uint8)*255
cv2.imwrite('combined_arm_enhanced.jpg',combined)
"""