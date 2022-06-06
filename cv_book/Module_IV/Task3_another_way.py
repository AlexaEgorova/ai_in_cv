import cv2
import numpy as np

from Module_I.load_calib import CalibReader
from Module_I.season_reader import SeasonReader
from Module_I.spatial_geometry_tools.calib import Calib
from Module_I.spatial_geometry_tools.camera import Camera
from Module_I.spatial_geometry_tools.point import Point3d as Point


class PointsCounter:

    def __init__(self, calib_dict, point_importance):
        self.prev_points = np.array([])
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.gps = []
        self.speed = 0
        self.left_2d_far = self.camera.project_point_3d_to_2d(Point((-1.5, 12, 0)))
        self.right_2d_far = self.camera.project_point_3d_to_2d(Point((1.5, 12, 0)))
        self.points_importance = point_importance  # Уровень отсечения точек
        self.yaw = 0
        self.common_diff = 0
        self.mult = 0


    def perv_points_projection_to_new(self, img):
        """Отрисовка точкек на изображении - старых и новых"""
        # Применение детектора Хариса для текущего изображения
        new_Harris = self.apply_Harris(img)

        time = 0.02 * 10 / 36  # Время для расчета пути (с переводом в м/с)

        # Работа с точками из предыдущего кадра
        if self.prev_points.size > 0:
            if self.mult == 0:
                self.mult = abs(np.sum(self.prev_points - new_Harris) / new_Harris.size) * 2000
                self.common_diff = np.ones(new_Harris.shape) * self.mult
            else:
                img[abs(self.prev_points - new_Harris) > self.common_diff] = [0, 0, 255]

        self.prev_points = new_Harris
        return img

    def apply_Harris(self, img):
        """
        Детектор Харриса.

        Принимает BGR изображение,
        возвращает изображение в GrayScale с отметкой углов.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.15)
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
        self.counter = PointsCounter(calib_dict, 0.20)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.counter.perv_points_projection_to_new(self.frame)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        self.counter.speed = self.shot[self._gps_name]['senseData']['speed']
        self.counter.yaw = self.shot[self._gps_name]['senseData']['yaw']
        return True

    def on_imu_frame(self):
        shot: dict = self.shot[self._imu_name]
        return True


if __name__ == '__main__':
    video_name = 'klt.427.003.mp4'
    init_args = {
        'path_to_data_root': '../data/tram/'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')
