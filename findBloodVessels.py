# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:32:40 2016

@author: queky
"""

import ridgeStrength
import findRidge
import numpy as np
import cv2

ridge = findRidge.findRidge()
ridge_strength = ridgeStrength.ridgeStrength()

#bin = np.logical_and(ridge,ridgeStrength)

#for i in range(np.size(bin,2)):
#    cv2.imwrite('output/findBloodVessels_results/vessels'+str(i)+'.jpg',bin[:,:,i].astype(np.uint8)*255)