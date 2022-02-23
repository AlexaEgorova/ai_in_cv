"""
Created on Wed Feb 23 20:00:49 2022

@author: egoro
"""

class Calib:
    def __init__(self, calib_dict):
        self.projection = calib_dict['K']
        self.distortion = calib_dict['D']

