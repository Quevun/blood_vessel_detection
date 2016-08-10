# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 07:05:33 2016

@author: queky
"""
import hlpr
import numpy as np


def normAxis2(cuboid):  #returns 3d array normalized along 3rd axis
    max_val = np.amax(abs(cuboid),2)  # max value for each 1st and 2nd axis coordinate
    max_val = max_val[:,:,np.newaxis]
    max_val = np.tile(max_val,(1,1,np.size(cuboid,2)))
    return cuboid/max_val

"""
cuboid = np.array([[[3,99,17],[15,337,945],[53,974,258]],[[47,2,97],[41,65,0],[41,57,62]],[[7,24,721],[67,25,44],[74,52,68]]])
cuboid2 = np.array([[[1,1,1],[1,0,1],[0,1,1]],[[1,1,1],[1,0,1],[0,0,1]],[[0,0,0],[1,0,0],[0,0,0]]])

pixel = hlpr.Pixel((2,1,0),1)
hlpr.Ridge.setCuboid(cuboid2)
ridge = hlpr.Ridge(pixel)
explored = ridge.growRidge()
"""

