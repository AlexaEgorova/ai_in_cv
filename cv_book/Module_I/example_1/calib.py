"""
Created on Wed Feb 23 20:00:49 2022

@author: egoro
"""

import numpy as np

class Calib:
    def __init__(self, calib_dict):
        self.cam_to_vr = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ])

        self.projection = calib_dict['K']
        print('projection:', self.projection, sep='\n', end='\n\n')
        self.fx = self.projection[0][0]
        self.cx = self.projection[0][-1]
        self.fy = self.projection[1][1]
        self.cy = self.projection[1][-1]

        self.distortion = calib_dict['D']
        print('distortion:', self.distortion, sep='\n', end='\n\n')

        angles = calib_dict['r']
        yaw = angles[0]
        pitch = angles[1]
        roll = angles[2]
        self.rotation = self.get_rotation_matrix_from_angles_vr([0], pitch*3.5, [0])
        self.rotation = self.cam_to_vr @ self.rotation
        print('rotation:', self.rotation, sep='\n', end='\n\n')

        self.translation = calib_dict['t']
        print('translation:', self.translation, sep='\n', end='\n\n')

    def rotation_matrix_from_angles(self, angles, order=(2, 0, 1)):
        """
        Args:
            angles: Углы поворота по осям x, y, z: в VR координатах это pitch, roll, yaw;
            order: порядок перемножения матриц. Порядок поворота на углы обратный. Умолчательный порядок - roll, pitch, yaw;
        """
        # cos = [np.cos(a) for a in angles]
        # sin = [np.sin(a) for a in angles]

        cos = []
        sin = []
        for a in angles:
            cos.append(np.cos(*a))
            sin.append(np.sin(*a))

        Rz = np.array([
            [cos[2], -sin[2], 0],
            [sin[2], cos[2], 0],
            [0, 0, 1],
        ])
        Ry = np.array([
            [cos[1], 0, sin[1]],
            [0, 1, 0],
            [-sin[1], 0, cos[1]],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, cos[0], -sin[0]],
            [0, sin[0], cos[0]],
        ])
        Rs = [Rx, Ry, Rz]

        R = Rs[order[0]] @ Rs[order[1]] @ Rs[order[2]]
        return R

    def get_rotation_matrix_from_angles_vr(self, yaw, pitch, roll):
        return self.rotation_matrix_from_angles([pitch, roll, yaw])

