"""
Created on Wed Feb 23 20:00:49 2022

@author: egoro
"""

import numpy as np

class Calib:
    def __init__(self, calib_dict):
        self.cam_to_vr = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])
        self.K = calib_dict['K']
        self.D = calib_dict['D']
        yaw, pitch, roll = calib_dict['r']
        self.r = self.rotation_matrix_from_angles([yaw, pitch, roll]).T
        self.t = calib_dict['t']

    def rotation_matrix_from_angles(self, angles):
        cosinuses, sinuses = [], []
        for a in angles:
            cosinuses.append(np.cos(*a))
            sinuses.append(np.sin(*a))
        Rx = np.array([
            [1, 0, 0],
            [0, cosinuses[0], -sinuses[0]],
            [0, sinuses[0], cosinuses[0]],
        ])
        Ry = np.array([
            [cosinuses[1], 0, sinuses[1]],
            [0, 1, 0],
            [-sinuses[1], 0, cosinuses[1]],
        ])
        Rz = np.array([
            [cosinuses[2], -sinuses[2], 0],
            [sinuses[2], cosinuses[2], 0],
            [0, 0, 1],
        ])

        R = Rz @ Ry @ Rx
        return R

