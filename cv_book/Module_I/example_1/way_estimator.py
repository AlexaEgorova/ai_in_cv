"""
Created on Wed Feb 23 20:02:37 2022

@author: egoro
"""

import numpy  as np
import cv2

from camera import Camera
from point import Point3d as Point

class WayEstimator:
    def __init__(self, camera: Camera):
        self.camera = camera
        # стаднартная ширина колеи - 1435 мм
        # near - по x в 0 метрах
        # far - по x в 10 в метрах
        self.left_3d_near = Point((-0.8, 1, 0))
        self.left_3d_far = Point((-0.8, 25, 0))
        self.right_3d_near = Point((0.8, 1, 0))
        self.right_3d_far = Point((0.8, 25, 0))

    def dray_way(self, img):
        left_2d_near = self.camera.project_point_3d_to_2d(self.left_3d_near)
        print('left_2d_near:', left_2d_near)

        left_2d_far = self.camera.project_point_3d_to_2d(self.left_3d_far)
        print('left_2d_far:', left_2d_far)

        right_2d_near = self.camera.project_point_3d_to_2d(self.right_3d_near)
        print('right_2d_near:', right_2d_near)

        right_2d_far = self.camera.project_point_3d_to_2d(self.right_3d_far)
        print('right_2d_far:', right_2d_far)

        cv2.line(img, right_2d_near, right_2d_far, (0, 0, 0), 5)
        cv2.line(img, left_2d_near, left_2d_far, (0, 0, 0), 5)

        return img

    def __det(self, a: np.array, b: np.array):
        return a[0] * b[1] - a[1] * b[0]

    def draw_coordinate_system(self, img):
        center3d = Point((0, 0, 0))
        center2d = self.camera.project_point_3d_to_2d(center3d)
        print("center2d:", center2d)
        cv2.circle(img, center2d, 5, (0, 0, 0), 5)

        for i in range(1, 20):
            x3d = Point((i, 0, 0))
            x2d = self.camera.project_point_3d_to_2d(x3d)
            print("x2d:", x2d)

            y3d = Point((0, i, 0))
            y2d = self.camera.project_point_3d_to_2d(y3d)
            print("y2d:", y2d)

            z3d = Point((0, 0, i))
            z2d = self.camera.project_point_3d_to_2d(z3d)
            print("z2d:", z2d)

            # blue x
            cv2.line(img, x2d, center2d, (255, 0, 0), 5)
            cv2.circle(img, x2d, 5, (255, 0, 0), 5)

            # green y
            cv2.line(img, y2d, center2d, (0, 255, 0), 5)
            cv2.circle(img, y2d, 5, (0, 255, 0), 5)

            # red z
            cv2.line(img, z2d, center2d, (0, 0, 255), 5)
            cv2.circle(img, z2d, 5, (0, 0, 255), 5)

        return img