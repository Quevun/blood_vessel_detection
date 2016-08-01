# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:32:40 2016

@author: queky
"""

import ridgeStrength
import findRidge
import numpy as np
import cv2

scale = np.arange(5,256,1)
lump_size = 1   #odd number
ridge = findRidge.findRidge(scale)
ridge_str_peak = ridgeStrength.ridgeStrength(scale,lump_size)

ridge2 = np.zeros(np.shape(ridge_str_peak))
for i in range(np.size(ridge_str_peak,2)):
    ridge2[:,:,i] = ridge[:,:,(lump_size-1)/2+i*lump_size]
    
bin = ridge2*ridge_str_peak

for i in range(np.size(bin,2)):
    cv2.imwrite('output/findBloodVessels_results/vessels'+str(i)+'.jpg',bin[:,:,i]*255)