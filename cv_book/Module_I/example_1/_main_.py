import cv2
from os.path import exists

from Module_I.example_calib.load_calib import CalibReader

from Module_I.example_1.way_estimator import WayEstimator
from Module_I.example_1.TrajectoryEstimator import TrajectoryEstimator

if __name__ == "__main__":
    par = ["K", "D", "r", "t"]
    calib_reader = CalibReader(
        file_name=r'../../data/tram/leftImage.yml',
        param=par)
    calib_dict = calib_reader.read()

    path = r'reels.bmp'
    img = cv2.imread(path)
    img = cv2.resize(img, (960, 540), interpolation = cv2.INTER_AREA) # todo: зачитать из калиба sz

    way_estimator = WayEstimator(calib_dict, 10)
    img = way_estimator.dray_way(img)

    # traj_estimator = TrajectoryEstimator(camera)
    # img = traj_estimator.dray_way(img, 1, 1, 0)

    cv2.namedWindow('reels')
    cv2.imshow('reels', img)
    cv2.waitKey(0)