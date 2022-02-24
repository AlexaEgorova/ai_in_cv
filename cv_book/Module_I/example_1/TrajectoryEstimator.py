import numpy  as np
import cv2

from camera import Camera

class  TrajectoryEstimator:
    def __init__(self, camera: Camera):
        self.camera = camera
        # стаднартная ширина колеи - 1435 мм
        # near - по x в 0 метрах
        # far - по x в 10 в метрах
        self.left_3d_near = np.array((0, 0, 0))
        self.left_3d_far = np.array((0, 0, 0))
        self.right_3d_near = np.array((0, 0, 0))
        self.right_3d_far = np.array((0, 0, 0))


    def dray_way(self, img, high, width, depth, ):
        left_down_2d_near = self.camera.project_point_3d_to_2d((0, 0, 0))
        left_down_2d_far = self.camera.project_point_3d_to_2d((depth, 0, 0))
        left_up_2d_near = self.camera.project_point_3d_to_2d((0, 0, high))
        left_up_2d_far = self.camera.project_point_3d_to_2d((depth, 0, high))
        right_up_2d_near = self.camera.project_point_3d_to_2d(0, width, high)
        right_up_2d_far = self.camera.project_point_3d_to_2d(depth, width, high)
        right_down_2d_near = self.camera.project_point_3d_to_2d(0, width, 0)
        right_down_2d_far = self.camera.project_point_3d_to_2d(depth, width, 0)

        cv2.rectangle(img, left_down_2d_near, right_up_2d_near, (255, 0, 0), 5)
        cv2.rectangle(img, left_down_2d_far, right_up_2d_far, (255, 0, 0), 5)
        cv2.line(img, left_down_2d_near, left_down_2d_far, (255, 0, 0), 5)
        cv2.line(img, left_up_2d_near, left_up_2d_far, (255, 0, 0), 5)
        cv2.line(img, right_down_2d_near, right_down_2d_far, (255, 0, 0), 5)
        cv2.line(img, right_up_2d_near, right_up_2d_far, (255, 0, 0), 5)

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