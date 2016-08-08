# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 07:05:33 2016

@author: queky
"""
import hlpr
import numpy as np

cuboid = np.array([[[3,99,17],[15,337,945],[53,974,258]],[[47,2,97],[41,65,0],[41,57,62]],[[7,24,721],[67,25,44],[74,52,68]]])
cuboid2 = np.array([[[1,1,1],[1,0,1],[0,1,1]],[[1,1,1],[1,0,1],[0,0,1]],[[0,0,0],[1,0,0],[0,0,0]]])

pixel = hlpr.Pixel((2,1,0),1)
hlpr.Ridge.setCuboid(cuboid2)
ridge = hlpr.Ridge(pixel)
explored = ridge.growRidge()