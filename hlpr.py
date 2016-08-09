# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:18:32 2016

@author: keisoku
"""
import cv2
import numpy as np
import copy

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
        
    def __mul__(self,other):
        assert other.cuboid.dtype == 'bool'
        return self.cuboid * other.cuboid
        
class Pixel(object):
    def __init__(self,coord,ridge_str):
        self.coord = coord
        self.ridge_str = ridge_str
        
    def getCoord(self):
        return (self.coord[0],self.coord[1],self.coord[2])
        
    def getRidgeStr(self):
        return self.ridge_str
        
class Ridge(object):
    #class_cuboid = None
    class_cuboid_mod = None
    
    @staticmethod
    def setCuboid(cuboid):
        #Ridge.class_cuboid = cuboid
        Ridge.class_cuboid_mod = copy.deepcopy(cuboid)
        
    @staticmethod
    def getCuboid():
        return Ridge.class_cuboid_mod
    
    def __init__(self,pixel):
        assert isinstance(pixel,Pixel)
        self.unexplored = [pixel]
        self.explored = {}
        Ridge.class_cuboid_mod[pixel.coord] = 0
        
    def checkAdjacent(self,pixel):
        y = pixel.coord[0]
        x = pixel.coord[1]
        t = pixel.coord[2]
        adjacentCoord = ((y-1,x,t),(y+1,x,t),(y,x-1,t),(y,x+1,t),(y,x,t-1),(y,x,t+1))
        for coord in adjacentCoord:
            if sum(np.array(coord) < 0) > 0:    # Don't check negative coordinates
                continue
            try:    # To deal with out of bound indices
                if Ridge.class_cuboid_mod[coord]:
                    self.unexplored.append(Pixel(coord,Ridge.class_cuboid_mod[coord]))
                    Ridge.class_cuboid_mod[coord] = 0 # Prevent from exploring same pixel again
            except IndexError:
                pass
        
    def growRidge(self):
        while self.unexplored:
            pixel = self.unexplored.pop()
            self.checkAdjacent(pixel)
            
            xy_coord = pixel.getCoord()[:2]
            if not xy_coord in self.explored:    # x,y coordinates as keys, list of pixels as values
                self.explored[xy_coord] = pixel
            else:
                if pixel.getRidgeStr() > self.explored[xy_coord].getRidgeStr():
                    self.explored[xy_coord] = pixel # replace pixel only if ridge strength is higher
            
            #self.explored.append(pixel)
            
            """# display list of explored and unexplored for testing purposes
            print 'unexplored:',
            for stuff in self.unexplored:
                print str(stuff.getCoord()),
            print ''
            print 'explored:',
            for more_stuff in self.explored:
                print str(more_stuff.getCoord()),
            print ''
            """
            
    def getTotalRidgeStr(self): # Calculate total ridge strength
        total_ridge_str = 0
        for xy_coord in self.explored:
            total_ridge_str += self.explored[xy_coord].getRidgeStr()
        return total_ridge_str
            
    def getImg(self):
        img = np.zeros((np.size(Ridge.class_cuboid_mod,0),np.size(Ridge.class_cuboid_mod,1)))
        for coord in self.explored:
            img[coord] = 255
        return img.astype(np.uint8)