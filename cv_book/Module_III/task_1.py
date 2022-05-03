from Module_I.season_reader import SeasonReader
from Module_I.load_calib import CalibReader
from Module_I.spatial_geometry_tools.calib import Calib
from Module_I.spatial_geometry_tools.camera import Camera
from Module_I.spatial_geometry_tools.point import Point3d as Point

import cv2
import numpy as np


class PointsCounter:

    def __init__(self, calib_dict):
        self.prev_points = np.array([])
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.gps = []
        self.speed = 0
        self.left_2d_far = self.camera.project_point_3d_to_2d(Point((-1.5, 12, 0)))
        self.right_2d_far = self.camera.project_point_3d_to_2d(Point((1.5, 12, 0)))

    def count_point_moving(self, Ri, tii1, yawi = 0, yawi1 = 0):
        """
        Рассмотрение точек в плоскости земли.

        Ri - точки на исходном кадре,
        Ti - начальная точка пути,
        tii1 - пройденный путь по GPS,
        yaw, yawi1 - исходный и следующий угол yaw по GPS
        """
        # Rz = np.array([
        #     [np.cos(yawi1 - yawi), -np.sin(yawi1 - yawi), 0],
        #     [np.sin(yawi1 - yawi), np.cos(yawi1 - yawi), 0],
        #     [0, 0, 1],
        # ], dtype=object)
        #
        # Ri1 = Ri @ Rz @ Ri
        # for x in Ri.shape[1]:
        #     for y in Ri.shape[0]:
        #         if(Ri[y, x] > 0):
        #             Ri[y, x] = self.camera.project_point_3d_to_2d(self.reproject_point_2d_to_3d_on_floor([x, y]) + tii1)

        RRi = []
        tii1 = np.array([[tii1[0]], [tii1[1]], [tii1[2]]])  # по идее, это вектор смещения
        for i in Ri:
            i.vec = i.vec + tii1    # смещение текущих точек
            a = self.camera.project_point_3d_to_2d(i)     # проецирование из 3d в 2d
            # проверка, что погрешность не выходит за рамки массива
            if a[0] < 540 and a[1] < 540:
                RRi.append(a)   # сохраняем точку
        return RRi

    def perv_points_projection_to_new(self, img):
        """Отрисовка точкек на изображении - старых и новых"""
        # Харис для текущего изображения
        new_Harris = self.apply_Harris(img)
        # время для расчета пути, идея бредовая не знаю откуда брать смещение
        time = 0.002
        # работа с точками с предыдущего кадра
        if self.prev_points.size > 0:
            # расчет перемещения точек с предыдущего кадра
            a = np.array(self.count_point_moving(self.get_3d_points_on_land(self.prev_points), [0, time * self.speed, 0. ]))
            # отрисовка рассчитанных точек синим
            img[a[:, 0], a[:, 1]] = [255, 0, 0]
            # отрисовака предыдущих точек зеленым
            img[self.prev_points > 0.01 * self.prev_points.max()] = [0, 255, 0]
        #отрисовка текущих точек красным
        img[new_Harris > 0.01 * new_Harris.max()] = [0, 0, 255]
        # сохранение текущих точек для следующего кадра
        self.prev_points = new_Harris

        return img

    def get_3d_points_on_land(self, new_Harris):
        """Функция отсечения точек не принадлежащих выбранному участку земли"""
        # отсечение выбранного участка на дороге
        new_Harris[:self.left_2d_far[1], :] = 0
        new_Harris[:, :self.left_2d_far[0]] = 0
        new_Harris[:, self.right_2d_far[0]:] = 0
        cv2.imshow('new Harris', new_Harris)

        # уровень важности точек
        pointsImportance = 0.01 * new_Harris.max()
        # получение координат точек проходящих по уровню
        points = np.argwhere(new_Harris > pointsImportance)
        # получение 3d координат точек
        points = np.apply_along_axis(self.reproject_point_2d_to_3d_on_floor, 1, points)

        return points

    @staticmethod
    def get_A_from_P_on_floor(P: np.ndarray) -> np.ndarray:
        """Значения первых двух столбцов неизменны, последние два столбца складываются"""
        h = 0 # процекция по земле, следовательно высота нулевая
        A = np.zeros((3, 3))
        A[0, 0], A[0, 1], A[0, 2] = P[0, 0], P[0, 1], h * P[0, 2] + P[0, 3]
        A[1, 0], A[1, 1], A[1, 2] = P[1, 0], P[1, 1], h * P[1, 2] + P[1, 3]
        A[2, 0], A[2, 1], A[2, 2] = P[2, 0], P[2, 1], h * P[2, 2] + P[2, 3]
        return A

#TODO: будет здорово, если какие-то матрицы можно предрасчитать и сохранить, тяжело для видео
    def reproject_point_2d_to_3d_on_floor(self, point2d: None):
        """Проецирование 2d точек, принадлежащих плоскости земли в 3d координаты"""
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

    def apply_Harris(self, img):
        """
        Детектор Харриса.

        Принимает BGR изображение,
        возвращает изображение в GrayScale с отметкой углов
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.15)
        # дилатация для отметки углов
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
        self.counter = PointsCounter(calib_dict)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.counter.perv_points_projection_to_new(self.frame)
        #countPoints.countPointMoving(np.array([50,-50]), 0, self.shot[])
        # self.counter.get_3d_points_on_land(self.frame)

        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        print(self.shot)         #[['yaw', 'timestamp', 'nord', 'west']])
        self.counter.speed = self.shot[self._gps_name]['senseData']['speed']
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


