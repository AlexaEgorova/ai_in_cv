"""
Created on Wed Feb 23 14:18:35 2022

@author: egoro
"""

import cv2
import numpy as np

class Calib:
    def __init__(self, projection):
        self.projection = projection


class Camera:
    def __init__(self, calib: Calib):
        self.calib = calib

    def project_point_3d_to_2d(self, point_3d: np.array):
        projection_result = self.calib.projection * point_3d
        if projection_result[2]:
            return np.array((projection_result[0] / projection_result[2],
                             projection_result[1] / projection_result[2]))
        return np.array((0, 0))


class WayEstimator:
    def __init__(self, camera: Camera):
        self.camera = camera
        # стаднартная ширина колеи - 1435 мм
        # near - по x в 0 метрах
        # far - по x в 10 в метрах
        self.left_3d_near = np.array((0, -0.7175, 0))
        self.left_3d_far = np.array((10, -0.7175, 0))
        self.right_3d_near = np.array((0, 0.7175, 0))
        self.right_3d_far = np.array((10, 0.7175, 0))

    def dray_way(self, img):
        left_2d_near = self.camera.project_point_3d_to_2d(self.left_3d_near)
        left_2d_far = self.camera.project_point_3d_to_2d(self.left_3d_far)
        right_2d_near = self.camera.project_point_3d_to_2d(self.right_3d_near)
        right_2d_far = self.camera.project_point_3d_to_2d(self.right_3d_far)

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
        
        
        
        

#class Camera:
#    """Класс с параметрами камеры."""
#    
#    def get_relative_pose():
#        pass
#
#
#class Pose:
#    """
#    Поза камеры.
#    
#    yaw - поворот вокруг оси z (рысканье)
#    pitch - поворот вокруг оси x (тангаж)
#    roll - поворот вокруг оси y (крен)
#    x, y, z - 3d координаты
#    
#    r - матрица поворота в данную систему координат (rotation)
#    t - координаты центра данной системы координат в исходной (translation)
#    """
#    def init(self, yaw: float, pitch: float, roll: float,
#             x: float, y: float, z: float):
#        self.yaw = yaw
#        self.pitch = pitch
#        self.roll = roll        
#        self.x = x
#        self.y = y
#        self.z = z
#        self.r = self.get_rotation_matrix()
#        self.t = np.array((x, y, z))
#        
#    def get_rotation_matrix(self):
#        pass
#    
#    def get_yaw_insensitive_pose():
#        pass
#    
#    def to_current_coordinate_system(self, point: np.array):
#        pass
#        
#
#class WayEstimator:
#    """
#    Класс работы с путём движения.
#    
#    right_unit_vector - единичный вектор вправо (по x)
#    forward_unit_vector - единичный вектор вперед (по y)
#    """
#    def __init__(self, camera: Camera):
#        self.camera = camera
#        self.right_unit_vector = np.array((1, 0, 0))
#        self.forward_unit_vector = np.array((0, 1, 0))
#    
#    def getHorizon(self, current_pose: Pose):
#        compensation_pose = current_pose.get_yaw_insensitive_pose()
#        compensation_pose.t = np.array((0, 0, 0))
#        self.forward_unit_vector = compensation_pose.to_current_coordinate_system(
#                self.forward_unit_vector)
#        self.right_unit_vector = compensation_pose.to_current_coordinate_system(
#                self.right_unit_vector)
#        origin = camera.get_relative_pose().t
#        way_direction = 3 * self.forward_unit_vector
#        
#        
#        
        
        
        
        