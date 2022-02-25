"""
Created on Wed Feb 23 20:01:56 2022

@author: egoro
"""

from example_1.calib import Calib
from example_1.point import Point3d  

class Camera:
    def __init__(self, calib: Calib):
        self.calib = calib

    def project_point_3d_to_2d(self, point3d: Point3d):
        translated = point3d.vec - self.calib.translation
        rotated = self.calib.rotation * translated

        X = rotated[0]
#        print('X:', X, end=' ')
        Y = rotated[1]
#        print('Y:', Y, end=' ')
        Z = rotated[2]
#        print('Z:', Z, end=' ')
        
        x = self.calib.fx*X + self.calib.cx*Z
        print('x:', x, end=' ')
        y = self.calib.fy*Y + self.calib.cy*Z
        print('y:', y, end=' ')
        w  = Z
        print('w:', w)
        
        res_2d = (x, y)
        if w > 0.001:
            res_2d = (int(x / w), int(y / w))
        
        res_2d = (res_2d[0], res_2d[1])
        return res_2d


