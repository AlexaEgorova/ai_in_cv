import cv2
from os.path import exists

from Module_I.example_calib.load_calib import CalibReader

from Module_I.example_1.calib import Calib
from Module_I.example_1.camera import Camera
from Module_I.example_1.way_estimator import WayEstimator
from Module_I.example_1.TrajectoryEstimator import TrajectoryEstimator

if __name__ == "__main__":
    par = ["K", "D", "r", "t"]
    calib_reader = CalibReader(
        file_name=r'../../data/tram/leftImage.yml',
        param=par)
    calib_dict = calib_reader.read()

    calib = Calib(calib_dict)
    camera = Camera(calib)
    way_estimator = WayEstimator(camera)
    traj_estimator = TrajectoryEstimator(camera)

    path = r'reels.bmp'
    img = cv2.imread(path)
    img = cv2.resize(img, (960, 540), interpolation = cv2.INTER_AREA) # todo: зачитать из калиба sz
    # b = cv2.drawFrameAxes(img, calib.projection, calib.distortion, calib.rotation, calib.translation, 1, 3)
    # b = way_estimator.draw_coordinate_system(img)

    # img = way_estimator.draw_coordinate_system(img)
    img = way_estimator.dray_way(img)
    # img = traj_estimator.dray_way(img, 1, 1, 0)
    cv2.namedWindow('reels')
    cv2.imshow('reels', img)
    cv2.waitKey(0)