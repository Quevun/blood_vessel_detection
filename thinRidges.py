# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 07:27:11 2016

@author: queky
"""
import cv2
import numpy as np
import skimage.morphology
import morphology
import anaFunc

def manualRemove(img):
    while not key == 13:
        cv2.imshow('Manual White Pixel Removal',img)
        cv2.setMouseCallback('image', anaFunc.getCoord)

img = np.load('output/eigen_arm_hori.npy')
skel = skimage.morphology.skeletonize(img>0)
#skel = np.random.rand(10,10)>0.6
branch_len = 20
pruned = skel.astype(np.uint8)*255
struc_ele1 = np.array([[-1,0,0],[1,1,0],[-1,0,0]])
struc_ele2 = np.array([[1,0,0],[0,1,0],[0,0,0]])
struc_ele_tuple = (struc_ele1,np.rot90(struc_ele1,1),np.rot90(struc_ele1,2),np.rot90(struc_ele1,3),
                   struc_ele2,np.rot90(struc_ele2,1),np.rot90(struc_ele2,2),np.rot90(struc_ele2,3))

for i in range(branch_len):
    end_points = np.zeros(np.shape(skel)).astype(np.bool)    
    for struc_ele in struc_ele_tuple:
        end_points = end_points + morphology.hitOrMiss(pruned,struc_ele)
    pruned = pruned * np.invert(end_points)

cv2.imwrite('output/pruned.jpg',pruned)
    
struc_ele3 = np.array([[0,0,0],[0,1,0],[0,0,0]])
single_points = morphology.hitOrMiss(pruned,struc_ele3)
pruned = pruned * np.invert(single_points)

cv2.imwrite('output/pruned_cleaned.jpg',pruned)