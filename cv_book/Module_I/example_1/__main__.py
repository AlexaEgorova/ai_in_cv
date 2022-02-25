# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:18:27 2022

@author: egoro
"""

import cv2
from os.path import exists

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
    
    path = 'example_1/reels.jpg'
    img = cv2.imread(path)
#    img = cv2.resize(img, (1920, 1080), interpolation = cv2.INTER_AREA)
    
    b = way_estimator.draw_coordinate_system(img)
    cv2.namedWindow('reels')
    cv2.imshow('reels', b)
    cv2.waitKey(0)

    