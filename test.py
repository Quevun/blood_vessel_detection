# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 07:05:33 2016

@author: queky
"""
import cv2

def getCoord(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print (x,y)