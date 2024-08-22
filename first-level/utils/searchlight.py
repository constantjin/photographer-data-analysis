# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:59:25 2021

@author: cms
"""

import numpy as np

class Searchlight:
    def __init__(self, radius):
        self.radius = radius
        
    
    def __str__(self):
        return 'Radius Size : %d' % (self.radius)
    
    
    def makeSphere(self):
        # make square box
        arr = np.full((2*self.radius+1, 2*self.radius+1, 2*self.radius+1), False)
        x0, y0, z0 = int(arr.shape[0]/2),int(arr.shape[1]/2),int(arr.shape[2]/2)
        
        # make sphere
        for x in range(x0-self.radius, x0+self.radius+1):
            for y in range(y0-self.radius, y0+self.radius+1):
                for z in range(z0-self.radius, z0+self.radius+1):
                    deb = np.linalg.norm(np.array([x0-x, y0-y, z0-z], dtype=np.float32))
                    arr[x,y,z] = True if deb <= self.radius else False
                    
        # return sphere coordinates 
        return arr

        
    def analysis(self, data, mask=None, post_func=None):
        self.data = data
        self.search_area = np.all(self.data, axis=3) if mask is None else mask
        self.post_func = (lambda x:x) if post_func is None else post_func
        self.sphere_mask = self.makeSphere()
        
        for x0 in range(self.radius, self.data.shape[0]-self.radius):
            for y0 in range(self.radius, self.data.shape[1]-self.radius):
                for z0 in range(self.radius, self.data.shape[2]-self.radius):
                    if self.search_area[x0, y0, z0]:
                        self.target = self.data[x0-self.radius:x0+self.radius+1, y0-self.radius:y0+self.radius+1, z0-self.radius:z0+self.radius+1].copy()
                        self.available_mask = np.all(self.target != 0, axis=3) & self.sphere_mask
                        
                        if np.any(self.available_mask):
                            values = self.target[self.available_mask]
                            result = self.post_func(values)
                            
                            yield result, (x0, y0, z0)