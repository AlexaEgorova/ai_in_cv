"""
Created on Wed Feb 23 14:18:35 2022

@author: egoro
"""

from example_calib.load_calib import CalibReader

from example_1.calib import Calib
from example_1.camera import Camera
from example_1.way_estimator import WayEstimator

if __name__ == "__main__":
    par = ["K", "D", "r", "t" ]
    calib_reader = CalibReader()
    calib_reader.initialize(file_name = '../data/tram/leftImage.yml', param = par)
    calib_dict = calib_reader.read()
    
    calib = Calib(calib_dict)
    camera = Camera(calib)
    way_estimator = WayEstimator(camera)