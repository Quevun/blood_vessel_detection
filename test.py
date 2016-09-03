# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 07:05:33 2016

@author: queky
"""
import hlpr
import numpy as np
import cv2
import morphology
import makeTrainSet


def normAxis2(cuboid):  #returns 3d array normalized along 3rd axis
    max_val = np.amax(abs(cuboid),2)  # max value for each 1st and 2nd axis coordinate
    max_val = max_val[:,:,np.newaxis]
    max_val = np.tile(max_val,(1,1,np.size(cuboid,2)))
    return cuboid/max_val
    
def enhanceRidges(img):
    temp = img**10
    temp2 = temp/np.amax(temp)*255
    return temp2

"""
cuboid = np.array([[[3,99,17],[15,337,945],[53,974,258]],[[47,2,97],[41,65,0],[41,57,62]],[[7,24,721],[67,25,44],[74,52,68]]])
cuboid2 = np.array([[[1,1,1],[1,0,1],[0,1,1]],[[1,1,1],[1,0,1],[0,0,1]],[[0,0,0],[1,0,0],[0,0,0]]])

pixel = hlpr.Pixel((2,1,0),1)
hlpr.Ridge.setCuboid(cuboid2)
ridge = hlpr.Ridge(pixel)
explored = ridge.growRidge()
"""

"""
img = cv2.imread('input/marker.bmp',cv2.IMREAD_GRAYSCALE)
scale = np.arange(1,30,1)
ridge_cuboid = findRidge(scale,img)
positive_Luv = ridge_cuboid > 0
sobelx = cv2.Sobel(positive_Luv[:,:,4].astype(np.int16),-1,1,0)
assert sobelx.dtype == np.int16
zero_cross = sobelx != 0
zero_cross = zero_cross.astype(np.uint8)*255
cv2.imshow('stuff',zero_cross)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""#test hlpr.scaleDerivZero
cuboid = np.random.rand(3,3,6)-0.5
zero_cross = hlpr.scaleDerivZero(cuboid)
"""

"""#skeleton
img = cv2.imread('output/eigen_results/arm_hori.jpg',0)
size = np.size(img)
skel = np.zeros(img.shape,np.uint8)
 
ret,img = cv2.threshold(img,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
 
while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
 
    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True
 
cv2.imshow("skel",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""# test getData.py
vessel_bin = np.random.rand(3,3) > 0.5
index = np.nonzero(vessel_bin)
sample_size = np.size(index[0])
v = [np.zeros((len(scales),5)) for _ in xrange(sample_size)]Ix = np.array([1,2,3,4,5])
######
Ix = np.random.rand(3,3)
Iy = np.random.rand(3,3)
Ixx = np.random.rand(3,3)
Iyy = np.random.rand(3,3)
Ixy = np.random.rand(3,3)
"""

"""
#test morphology.py
img = (np.random.rand(10,10)>0.8).astype(np.uint8)*255
#img = np.array([[0,0,0],[255,255,0],[0,0,0]])
#img = np.load('output/test.npy')
struc_ele = np.array([[-1,0,0],[1,1,0],[-1,0,0]])
hit_or_miss = morphology.hitOrMiss(img,struc_ele)
print img
print ''
print hit_or_miss.astype(np.uint8)*255
"""

"""# test multiple waitKey()
img = cv2.imread('input/IR3/test7.bmp',0)
img2 = cv2.imread('input/IR3/test5.bmp',0)
cv2.imshow('stuff',img)
key = cv2.waitKey(1000)
cv2.imshow('stuff',img2)
key = cv2.waitKey(1000)
cv2.destroyAllWindows()
"""

"""# test makeTrainSet.py
Ix = (np.random.rand(3,3)*100).astype(np.uint8)
Iy = (np.random.rand(3,3)*100).astype(np.uint8)
Ixx = (np.random.rand(3,3)*100).astype(np.uint8)
Iyy = (np.random.rand(3,3)*100).astype(np.uint8)
Ixy = (np.random.rand(3,3)*100).astype(np.uint8)

index = (np.array([0,1,2]),np.array([0,2,1]))
"""

"""# test randNonVessel
shape = (4L,4L)
vessel_index = (np.array([0,0,1,1,2,2]),np.array([1,2,0,2,0,1]))
foo = makeTrainSet.randNonVessel(shape,vessel_index)
"""