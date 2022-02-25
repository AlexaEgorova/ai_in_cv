"""
Created on Wed Feb 23 20:00:49 2022

@author: egoro
"""

class Calib:
    def __init__(self, calib_dict):
        self.projection = calib_dict['K']
        print('projection:', self.projection, sep='\n', end='\n\n')
        self.fx = self.projection[0][0]
        self.cx = self.projection[0][-1]
        self.fy = self.projection[1][1]
        self.cy = self.projection[1][-1]
        
        self.distortion = calib_dict['D']
        print('distortion:', self.distortion, sep='\n', end='\n\n')
        
        self.rotation = calib_dict['r']
        print('rotation:', self.rotation, sep='\n', end='\n\n')
        print(type(self.rotation))
        
        self.translation = calib_dict['t']
        print('translation:', self.translation, sep='\n', end='\n\n')
        

