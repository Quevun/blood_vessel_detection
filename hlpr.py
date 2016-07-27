# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:18:32 2016

@author: keisoku
"""
import cv2
import numpy as np

def getScaleSpace(img,scale):
    sigma = np.sqrt(scale)
    size = (np.ceil(sigma)*6+1).astype(int)
    scaled_img = []
    for i in range(len(sigma)):
        scaled_img.append(cv2.GaussianBlur(img,(size[i],size[i]),sigma[i]))
    return scaled_img
    
def getScaledImg(img,scale):
    sigma = np.sqrt(scale)
    size = int(np.ceil(sigma)*6+1)
    img = img.astype(np.float64)
    scaled_img = cv2.GaussianBlur(img,(size,size),sigma)
    #scaled_img = cv2.medianBlur(img,size)    
    return scaled_img

def float2uint(sobelx):
    import numpy as np
    sobelx = sobelx - np.amin(sobelx,(0,1))
    sobelx = sobelx / np.amax(sobelx,(0,1))
    sobelx =  sobelx * 255
    sobelx = sobelx.astype(np.uint8)
    return sobelx
    
class ScaledImage:
    def __init__(self,img,scale):
        self.scale = scale
        self.img = getScaledImg(img,scale) # floating point image
        self.sobelx = None
        self.sobely = None
        self.sobelxx = None
        self.sobelyy = None
        self.sobelxy = None
        
    def getImg(self):
        return self.img
        
    def getScale(self):
        return self.scale
        
    def getSobelx(self):
        if self.sobelx is None:
            self.sobelx = cv2.Sobel(self.img,cv2.CV_64F,1,0)
            return self.sobelx
        else:
            return self.sobelx
            
    def getSobely(self):
        if self.sobely is None:
            self.sobely = cv2.Sobel(self.img,cv2.CV_64F,0,1)
            return self.sobely
        else:
            return self.sobely
        
    def getSobelxx(self):
        scale = self.getScale()
        if self.sobelxx is None:
            self.sobelxx = cv2.Sobel(self.img,cv2.CV_64F,2,0)#,ksize=scale + scale % 2 - 1)
            return self.sobelxx
        else:
            return self.sobelxx
            
    def getSobelyy(self):
        scale = self.getScale()
        if self.sobelyy is None:
            self.sobelyy = cv2.Sobel(self.img,cv2.CV_64F,0,2)#,ksize=scale + scale % 2 - 1)
            return self.sobelyy
        else:
            return self.sobelyy
            
    def getSobelxy(self):
        scale = self.getScale()
        if self.sobelxy is None:
            self.sobelxy = cv2.Sobel(self.img,cv2.CV_64F,1,1)#,ksize=scale + scale % 2 - 1)
            return self.sobelxy
        else:
            return self.sobelxy
            
    def findRidge(self):
        Lx = self.getSobelx()
        Ly = self.getSobely()
        Lxy = self.getSobelxy()
        Lxx = self.getSobelxx()
        Lyy = self.getSobelyy()
        
        temp = (Lxx - Lyy)/np.sqrt((Lxx-Lyy)**2 + 4*Lxy**2)
        sin_beta = np.sign(Lxy) * np.sqrt((1-temp)/2)
        cos_beta = np.sqrt((1+temp)/2)
        
        Lp = sin_beta * Lx - cos_beta * Ly  # first derivatives of principal directions
        Lq = cos_beta * Lx + sin_beta * Ly  # first derivatives of principal directions
        
        Lpp = sin_beta**2*Lxx - 2*sin_beta*cos_beta*Lxy - cos_beta**2*Lyy
        Lqq = cos_beta**2*Lxx + 2*sin_beta*cos_beta*Lxy + sin_beta**2*Lyy
        
        bin1 = Lq.astype(np.int32) == 0
        bin2 = Lqq >= 0.005
        bin3 = abs(Lqq) >= abs(Lpp)
        bin4 = np.logical_and(bin3,np.logical_and(bin1,bin2))
        bin4 = bin4 == False
        ridge = self.getImg() * bin4
        return ridge
        
        ######################################################
        # Threshold with Lpp Lqq
        #eigen = np.zeros((np.size(Lpp,0),np.size(Lpp,1),2))
        #eigen[:,:,0] = Lpp
        #eigen[:,:,1] = Lqq        
        #major = np.amax(eigen,2) > 0.05
        #major = major.astype(np.uint8)*255
        #major = Lqq > 0.05
        #major = major.astype(np.uint8)*255      
        #return major
        ######################################################
            
    def getRidgeStrength(self):
        scale = self.getScale()
        sobelxx = self.getSobelxx()
        sobelyy = self.getSobelyy()
        sobelxy = self.getSobelxy()
        return scale**4 * (sobelxx + sobelyy)**2 * ((sobelxx - sobelyy)**2 + 4 * sobelxy**2)