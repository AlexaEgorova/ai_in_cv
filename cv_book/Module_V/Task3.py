import math

import cv2
from Module_I.season_reader import SeasonReader
from Module_I.load_calib import CalibReader
import numpy as np


class FindHorizont:
    def __init__(self):
        self.line = []
        self.intersection_points = {}

    def line_intersection(self, line1, line2):
        xdiff = (line1[0] - line1[2], line2[0] - line2[2])
        ydiff = (line1[1] - line1[3], line2[1] - line2[3])  # Typo was here

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det((line1[0], line1[1]), (line1[2], line1[3])), det((line2[0], line2[1]), (line2[2], line2[3])))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return math.ceil(x), math.ceil(y)

    def belong_to_line(self, x1, y1, x2, y2, x3, y3):
        if y2 == y1 and y2 == y3:
            return True
        if x1 == x2 & x1 == x3:
            return True

        return (x1 - x3) * (y2 - y1) == (x2 - x1) * (y1 - y3)

    def find_lines(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50,
                          200)  # Find the edges in the image using canny detector  # Convert the image to gray-scale
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=10,
                                maxLineGap=250).squeeze()  # поиск линий с помощью Хаффа
        roads = lines[(lines[:, 0] < 650) & (lines[:, 0] > 300) & (lines[:, 2] < 650) & (
                lines[:, 2] > 300)]  # отсеивание линий, не принадлежащих рельсам
        for i in roads:
            for j in roads:
                if i.all() != j.all():
                    x, y = self.line_intersection(i, j)  # поиск точки пересечения линий
                    if (x >= 0 & x < img.shape[1] & y >= 0 & y < img.shape[
                        0]):  # отсеиваем точки, не принадлежащие изображению
                        self.intersection_points[(x, y)] = self.intersection_points.setdefault((x, y),
                                                                                               0) + 1  # счиатем сколько прямых пересеклось в этой точке
                        cv2.line(img, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 3)  # рисуем линии
                        cv2.line(img, (j[0], j[1]), (j[2], j[3]), (255, 0, 0), 3)  # рисуем точку пересечения
        if len(self.intersection_points):
            point = max(self.intersection_points,
                        key=self.intersection_points.get)  # находим точку, в которой больше всего пересечений
            cv2.circle(img, [point[0], point[1]], 5, (0, 255, 0))  # рисуем точку пересечения
            horizonts = []
            for i in lines:
                if self.belong_to_line(i[0], i[1], i[2], i[3], point[0], point[
                    1]):  # линии, проходящие через точку пересечения, претенденты на линию горизонта
                    horizonts.append(i)  # если линия найдена, то сохраняем
            if len(horizonts) != 0:  #
                # print(min(self.intersection_points, key=abs(self.intersection_points[1] - self.intersection_points[3])))
                self.line = horizonts[0]
        if len(self.line) == 4:  # отрисовываем сохраненную
            cv2.line(img, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (0, 0, 255), 3)


class Reader(SeasonReader):
    """Обработка видеопотока."""

    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/tram/leftImage.yml',
            param=par)
        self.horizont = FindHorizont()

        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.horizont.find_lines(self.frame)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
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
