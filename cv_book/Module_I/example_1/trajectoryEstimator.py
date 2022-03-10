
import numpy as np
import cv2

from Module_I.example_1.calib import Calib
from Module_I.example_1.camera import Camera
from Module_I.example_1.point import Point3d as Point

class TrajectoryEstimator:
    def __init__(self, calib_dict: np.array, high, wight, depth, length: int):  # задаем калиб, высоту, ширину, глебину параллелограмма и расстояние от камеры
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.left_3d_near1 = Point((-wight/2, length, 0))
        self.left_3d_far1 = Point((-wight/2, depth, 0))
        self.right_3d_near1 = Point((wight/2, length, 0))
        self.right_3d_far1 = Point((wight/2, depth, 0))
        self.left_3d_near_up1 = Point((-wight/2, length, high))
        self.left_3d_far_up1 = Point((-wight/2, depth, high))
        self.right_3d_near_up1 = Point((wight/2, length, high))
        self.right_3d_far_up1 = Point((wight/2, depth, high))

    def dray_way(self, img):
        left_2d_near1 = self.camera.project_point_3d_to_2d(self.left_3d_near1)
        left_2d_far1 = self.camera.project_point_3d_to_2d(self.left_3d_far1)
        right_2d_near1 = self.camera.project_point_3d_to_2d(self.right_3d_near1)
        right_2d_far1 = self.camera.project_point_3d_to_2d(self.right_3d_far1)
        left_2d_near_up1 = self.camera.project_point_3d_to_2d(self.left_3d_near_up1)
        left_2d_far_up1 = self.camera.project_point_3d_to_2d(self.left_3d_far_up1)
        right_2d_near_up1 = self.camera.project_point_3d_to_2d(self.right_3d_near_up1)
        right_2d_far_up1 = self.camera.project_point_3d_to_2d(self.right_3d_far_up1)

        black_color = (0, 0, 0)
        line_width = 5

        cv2.rectangle(img, left_2d_near1, right_2d_near1, black_color, line_width)
        cv2.rectangle(img, left_2d_far1, right_2d_far1, black_color, line_width)
        cv2.rectangle(img, left_2d_far_up1, right_2d_far_up1, black_color, line_width)
        cv2.rectangle(img, left_2d_near_up1, right_2d_near_up1, black_color, line_width)
        cv2.rectangle(img, left_2d_far_up1, left_2d_far1, black_color, line_width)
        cv2.rectangle(img, right_2d_far_up1, right_2d_far1, black_color, line_width)
        cv2.rectangle(img, left_2d_near_up1, left_2d_near1, black_color, line_width)
        cv2.rectangle(img, right_2d_near_up1, right_2d_near1, black_color, line_width)
        cv2.line(img, left_2d_near1, left_2d_far1, black_color, line_width)
        cv2.line(img, left_2d_near_up1, left_2d_far_up1, black_color, line_width)
        cv2.line(img, right_2d_near1, right_2d_far1, black_color, line_width)
        cv2.line(img, right_2d_near_up1, right_2d_far_up1, black_color, line_width)

        return img