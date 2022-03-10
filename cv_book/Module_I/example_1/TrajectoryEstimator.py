
import numpy as np
import cv2

from Module_I.example_1.calib import Calib
from Module_I.example_1.camera import Camera
from Module_I.example_1.point import Point3d as Point

class TrajectoryEstimator:
    def __init__(self, calib_dict: np.array, ways_length: int):
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.left_3d_near1 = Point((-0.8, 10, 0))
        self.left_3d_far1 = Point((-0.8, ways_length - 1, 0))
        self.right_3d_near1 = Point((0.8, 10, 0))
        self.right_3d_far1 = Point((0.8, ways_length - 1, 0))
        self.left_3d_near_up1 = Point((-0.8, 10, 1))
        self.left_3d_far_up1 = Point((-0.8, ways_length - 1, 1))
        self.right_3d_near_up1 = Point((0.8, 10, 1))
        self.right_3d_far_up1 = Point((0.8, ways_length - 1, 1))

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

    def __det(self, a: np.array, b: np.array):
        return a[0] * b[1] - a[1] * b[0]

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

