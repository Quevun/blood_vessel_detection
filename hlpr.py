# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:18:32 2016

@author: keisoku
"""
import cv2
import numpy as np

def getScaleSpace(img,scale):
    sigma = np.sqrt(scale)
    size = (np.ceil(sigma)*10+1).astype(int)
    scaled_img = []
    for i in range(len(sigma)):
        scaled_img.append(cv2.GaussianBlur(img,(size[i],size[i]),sigma[i]))
    return scaled_img
    
def getScaledImg(img,scale):
    sigma = np.sqrt(scale)
    size = int(np.ceil(sigma)*10+1)
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
    
class ScaledImage(object):
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
        if self.sobelxx is None:
            self.sobelxx = cv2.Sobel(self.img,cv2.CV_64F,2,0)#,ksize=scale + scale % 2 - 1)
            return self.sobelxx
        else:
            return self.sobelxx
            
    def getSobelyy(self):
        if self.sobelyy is None:
            self.sobelyy = cv2.Sobel(self.img,cv2.CV_64F,0,2)#,ksize=scale + scale % 2 - 1)
            return self.sobelyy
        else:
            return self.sobelyy
            
    def getSobelxy(self):
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
        bin2 = Lqq >= 0.05
        bin3 = abs(Lqq) >= abs(Lpp)
        bin4 = np.logical_and(bin3,np.logical_and(bin1,bin2))
        #ridge = self.getImg() * bin4
        return bin4
            
    def getRidgeStrength(self):
        scale = self.getScale()
        sobelxx = self.getSobelxx()
        sobelyy = self.getSobelyy()
        sobelxy = self.getSobelxy()
        return scale**3 * (sobelxx + sobelyy)**2 * ((sobelxx - sobelyy)**2 + 4 * sobelxy**2)

class RidgeStrCuboid(object):
    def __init__(self,img,scale):
        self.shape = (np.size(img,0),np.size(img,1),len(scale))
        self.cuboid = np.zeros((np.size(img,0),np.size(img,1),len(scale)))
        for i in range(len(scale)):
            self.cuboid[:,:,i] = ScaledImage(img,scale[i]).getRidgeStrength()
            
    def getScaleDeriv(self):
        max_i = self.shape[2]-1
        self.scale_deriv = np.zeros(self.shape)
        for i in range(self.shape[2]):
            self.scale_deriv[:,:,i] = self.cuboid[:,:,(i+1)%max_i]-self.cuboid[:,:,i-1]
        return self.scale_deriv
        
    def getScaleDeriv2(self):
        max_i = self.shape[2] - 1
        self.scale_deriv2 = np.zeros(self.shape)
        if not hasattr(self,'scale_deriv'):
            print "First order scale derivative doesn't exist, creating one..."
            self.getScaleDeriv()
        for i in range(self.shape[2]):    
            self.scale_deriv2[:,:,i] = self.scale_deriv[:,:,(i+1)%max_i]-self.scale_deriv[:,:,i-1]
        return self.scale_deriv2
            
class BinImgCuboid(object):
    def __init__(self,cuboid):
        assert cuboid.dtype == 'bool'
        self.cuboid = cuboid
        self.shape = np.shape(cuboid)
        
    def lump(self,lump_size):
        lump_bin = 
        for i in range(0,self.shape[2]-lump_size+1,lump_size):
            lump_bin = self.cuboid[:,:,i]
            for j in range(1,lump_size):
                lump_bin += self.cuboid[:,:,i+j]
                
        return BinImgCuboid()