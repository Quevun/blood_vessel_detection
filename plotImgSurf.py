# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 14:27:44 2016

@author: keisoku
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import hlpr

def plotImg(img):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = range(np.size(img,1))
    y = range(np.size(img,0))
    X, Y = np.meshgrid(x, y)
    plt.gca().invert_yaxis()
    ax.plot_surface(X,Y,img)
    
img = cv2.imread('input/test.bmp',cv2.IMREAD_GRAYSCALE)
img = cv2.pyrDown(img)
img = hlpr.getScaledImg(img,225)
plotImg(img)