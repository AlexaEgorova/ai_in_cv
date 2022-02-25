"""
Created on Wed Feb 23 20:02:37 2022

@author: egoro
"""

import numpy  as np
import cv2

from example_1.camera import Camera
from example_1.point import Point3d as Point

class WayEstimator:
    def __init__(self, camera: Camera):
        self.camera = camera
        # стаднартная ширина колеи - 1435 мм
        # near - по x в 0 метрах
        # far - по x в 10 в метрах
        self.left_3d_near = Point((0, -0.7175, 1))
        self.left_3d_far = Point((10, -0.7175, 1))
        self.right_3d_near = Point((0, 0.7175, 1))
        self.right_3d_far = Point((10, 0.7175, 1))

    def dray_way(self, img):
        left_2d_near = self.camera.project_point_3d_to_2d(self.left_3d_near)
        print('left_2d_near:', left_2d_near)
        
        left_2d_far = self.camera.project_point_3d_to_2d(self.left_3d_far)
        print('left_2d_far:', left_2d_far)
        
        right_2d_near = self.camera.project_point_3d_to_2d(self.right_3d_near)
        print('right_2d_near:', right_2d_near)
        
        right_2d_far = self.camera.project_point_3d_to_2d(self.right_3d_far)
        print('right_2d_far:', right_2d_far)

        intersection_pt = self.intersect_lines(left_2d_near, left_2d_far,
                                               right_2d_near, right_2d_far)
        
        cv2.line(img, left_2d_near, intersection_pt, (255, 0, 0), 5)
        cv2.line(img, right_2d_near, intersection_pt, (255, 0, 0), 5)

    def __det(self, a: np.array, b: np.array):
        return a[0]*b[1] - a[1]*b[0]

    def intersect_lines(self,
                        pt1_of_line1: np.array, pt2_of_line1: np.array,
                        pt1_of_line2: np.array, pt2_of_line2: np.array):
        line1 = np.array([pt1_of_line1, pt2_of_line1])
        line2 = np.array([pt1_of_line2, pt2_of_line2])
        x_diff = line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]
        y_diff = line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]

        div = self.__det(x_diff, y_diff)
        if div == 0:
            print("Lines do not intersect")
            return np.array((0, 0))
        d = (self.__det(*line1), self.__det(*line2))
        x = self.__det(d, x_diff)
        y = self.__det(d, y_diff)
        return np.array((x, y))
    
    def draw_coordinate_system(self, img):
        center3d = Point((0, 0, 0))
        center2d = self.camera.project_point_3d_to_2d(center3d)
        print("center2d:", center2d)
        cv2.circle(img, center2d, 5, (0, 0, 0), 5)

        for i in range(1, 5):
            x3d = Point((i, 0, 0))
            x2d = self.camera.project_point_3d_to_2d(x3d)
            print("x2d:", x2d)
            
            y3d = Point((0, i, 0))
            y2d = self.camera.project_point_3d_to_2d(y3d)
            print("y2d:", y2d)
            
            z3d = Point((0, 0, i))
            z2d = self.camera.project_point_3d_to_2d(z3d)
            print("z2d:", z2d)
                
    #        cv2.line(img, (206, 1000), center_2d, (0, 0, 0), 5)
    #        cv2.line(img, (842, 1000), center_2d, (0, 0, 0), 5)
            cv2.line(img, x2d, center2d, (255, 0, 0), 5)
            cv2.circle(img, x2d, 5, (255, 0, 0), 5)
            cv2.line(img, y2d, center2d, (0, 255, 0), 5)
            cv2.circle(img, y2d, 5, (0, 255, 0), 5)
            cv2.line(img, z2d, center2d, (0, 0, 255), 5)
            cv2.circle(img, z2d, 5, (0, 0, 255), 5)
    
        return img
