"""
Created on Wed Feb 23 20:01:56 2022

@author: egoro
"""

import numpy as np

from example_1.calib import Calib

class Camera:
    def __init__(self, calib: Calib):
        self.calib = calib

    def project_point_3d_to_2d(self, point_3d: np.array):
        projection_result = self.calib.projection * point_3d
        if projection_result[2]:
            return np.array((projection_result[0] / projection_result[2],
                             projection_result[1] / projection_result[2]))
        return np.array((0, 0))

