import cv2
from Module_I.season_reader import SeasonReader
from Module_I.load_calib import CalibReader
import numpy as np


class FindDriver:
    def __init__(self):
        self.prev_points_kp = np.array([])
        self.prev_points_desc = np.array([])
        self.prev_img = np.array([])

    def sift_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()                        # pip install opencv-contrib-python
        # kp is the keypoints
        #
        # desc is the SIFT descriptors, they're 128-dimensional vectors
        # that we can use for our final features
        kp, desc = sift.detectAndCompute(gray, None)
        # img_with_points = cv2.drawKeypoints(gray, kp, img.copy())
        # cv2.imshow("points", img_with_points)
        return kp, desc

    def find_match(self, img):

        img_with_points_kp, img_with_points_desc = self.sift_features(img)
        if self.prev_points_kp != [] and self.prev_points_desc != []:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(self.prev_points_desc, img_with_points_desc)
        # Sort the matches in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
        # draw the top N matches
            N_MATCHES = 20
            match_img = cv2.drawMatches(
                img, img_with_points_kp,
                self.prev_img, self.prev_points_kp,
                matches[:N_MATCHES], img.copy(), flags=0)
            cv2.imshow("match", match_img)

        self.prev_points_kp = img_with_points_kp
        self.prev_points_desc = img_with_points_desc
        self.prev_img = img



class Reader(SeasonReader):
    """Обработка видеопотока."""

    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/tram/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        self.findDriver = FindDriver()
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.findDriver.find_match(self.frame)

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
