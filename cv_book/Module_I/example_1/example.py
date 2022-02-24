"""
Created on Wed Feb 23 14:18:35 2022

@author: egoro
"""

import cv2
from season_reader import SeasonReader
from way_estimator import WayEstimator
from camera import Camera
from calib import Calib
from Module_I.example_calib.load_calib import CalibReader


class MyReader(SeasonReader):
    def on_init(self):
        par = ["K", "D", "r", "t"]
        calib_reader = CalibReader(file_name=r'C:\Users\Dns\OneDrive - НИТУ МИСиС\Документы\Учеба\Магистратура\2 семестр\OpenCV\ai_in_cv\cv_book\data\tram\leftImage.yml', param=par)
        calib_dict = calib_reader.read()

        calib = Calib(calib_dict)
        camera = Camera(calib)
        self.ways = WayEstimator(camera)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f"GrabMsec: {self.frame_grab_msec}", (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.ways.dray_way(self.frame)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        return True

    def on_imu_frame(self):
        shot: dict = self.shot[self._imu_name]
        return True


if __name__ == "__main__":

    init_args = {
        'path_to_data_root': '../../data/tram/'
    }
    s = MyReader()
    s.initialize(**init_args)
    s.run()
    print("Done!")

# if __name__ == "__main__":
# par = ["K", "D", "r", "t"]
# calib_reader = CalibReader(file_name=r'C:\Users\Dns\OneDrive - НИТУ МИСиС\Документы\Учеба\Магистратура\2 семестр\OpenCV\ai_in_cv\cv_book\data\tram\leftImage.yml', param=par)
# calib_dict = calib_reader.read()
#
# calib = Calib(calib_dict)
# camera = Camera(calib)
# way_estimator = WayEstimator(camera)
