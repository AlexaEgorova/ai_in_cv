import cv2
import numpy as np
from Module_I.season_reader import SeasonReader
from Module_I.load_calib import CalibReader
from Module_I.spatial_geometry_tools.calib import Calib
from Module_I.spatial_geometry_tools.camera import Camera
from Module_I.spatial_geometry_tools.point import Point3d as Point

class countPoints:

    def __init__(self, calib_dict):
        self.prev_points = np.ndarray(list())
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)

    # Ri - точки на исходном кадре, Ti - начальная точка пути, tii1 - пройденный путь по GPS, yaw, yawi1 - исходный и следующий угол yaw по GPS
    def countPointMoving(self, Ri, Ti, tii1, yawi, yawi1): #рассмотреть только точки в плоскости дороги(плоскость земля)

        # Rz = np.array([
        #     [np.cos(yawi1 - yawi), -np.sin(yawi1 - yawi), 0],
        #     [np.sin(yawi1 - yawi), np.cos(yawi1 - yawi), 0],
        #     [0, 0, 1],
        # ], dtype=object)
        #
        # Ri1 = Ri @ Rz @ Ri
        Ti1: object = Ti + Ri * tii1    #Ri1
        print(Ti1)
        return Ti1

    #отрисовывает точки на изображении, старые и новые
    def perv_points_pojection_to_new(self, img):
        new_Harris = self.apply_Harris(img)
        # Бинаризация для контроля количества точек, может варьироваться в зависимости от задачи
        img[new_Harris > 0.01 * new_Harris.max()] = [0, 0, 255]
        img[self.prev_points > 0.01 * self.prev_points.max()] = [0, 255, 0]
        self.prev_points = new_Harris
        img[239:240] = [255, 0, 0]
        return img

    def get_3d_points_on_land(self, img):
        left_3d_near = Point((0, 4, 0))
        left_3d_far = Point((0, 5, 0))
        left_2d_near = self.camera.project_point_3d_to_2d(left_3d_near)
        left_2d_far = self.camera.project_point_3d_to_2d(left_3d_far)
        #проверка, что преобразования возвращают одно и то же
        print(self.project_point_2d_to_3d([453, 530]))
        print(left_2d_near)
        print(self.project_point_2d_to_3d([458, 470]))
        print(left_2d_far)
        return img

    @staticmethod
    def get_A_from_P_on_floor(P: np.ndarray) -> np.ndarray:
        """Значения первых двух столбцов неизменны, последние два столбца складываются"""
        h = 0 # процекция по земле, следовательно высота нулевая
        A = np.zeros((3, 3))
        A[0, 0], A[0, 1], A[0, 2] = P[0, 0], P[0, 1], h * P[0, 2] + P[0, 3]
        A[1, 0], A[1, 1], A[1, 2] = P[1, 0], P[1, 1], h * P[1, 2] + P[1, 3]
        A[2, 0], A[2, 1], A[2, 2] = P[2, 0], P[2, 1], h * P[2, 2] + P[2, 3]
        return A

    def reproject_point_2d_to_3d_on_floor(self, point2d: None):
        if point2d is None:
            point2d = []
        h = 0 # процекция по земле, следовательно высота нулевая
        R = self.calib.cam_to_vr @ self.calib.r # меняем местами оси
        affine_matrix = np.concatenate((R, -R @ self.calib.t), 1)
        P = self.calib.K @ affine_matrix
        A = self.get_A_from_P_on_floor(P)
        A_inv = np.linalg.inv(A)
        p_ = A_inv @ Point((point2d[0], point2d[1], 1)).vec
        reprojected = Point((p_[0] / p_[2], p_[1] / p_[2], h))
        return reprojected


    #Детектор Хариса, принимает BGR изображение, возвращает изображение в GrayScale с отметкой углов и
    def apply_Harris(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.15)
        #дилатация для отметки углов
        dst = cv2.dilate(dst, None)
        return dst


class Reader(SeasonReader):
    """Обработка видеопотока."""
    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/tram/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        self.counter = countPoints(calib_dict)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        #countPoints.countPointMoving(np.array([50,-50]), 0, self.shot[])
        self.counter.get_3d_points_on_land(self.frame)
        #self.counter.perv_points_pojection_to_new(self.frame)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        # print(self.shot[self._gps_name]['senseData']['yaw'])         #[['yaw', 'timestamp', 'nord', 'west']])
        # print('--------')
        return True

    def on_imu_frame(self):
        shot: dict = self.shot[self._imu_name]
        return True


if __name__ == '__main__':
    init_args = {
        'path_to_data_root': '../data/tram/'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')


