import cv2
from season_reader import SeasonReader
from Module_I.example_1.trajectoryEstimator import TrajectoryEstimator
from Module_I.example_calib.load_calib import CalibReader


class MyReader2(SeasonReader):
    def on_init(self):
        par = ["K", "D", "r", "t"]
        calib_reader = CalibReader(
            file_name=r'../../data/tram/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        self.traj_estimator = TrajectoryEstimator(calib_dict, 1, 1, 25, 8)

        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f"GrabMsec: {self.frame_grab_msec}", (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.traj_estimator.dray_way(self.frame)

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
    s = MyReader2()
    s.initialize(**init_args)
    s.run()
    print("Done!")

