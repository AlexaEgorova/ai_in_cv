import cv2
import numpy as np


class LightsStruckReducer:
    """Класс для приглушения засветки от фар"""

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
        look_up_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(img, look_up_table)
        return res

    @staticmethod
    def get_map_mask(img, radius):
        """Получение маски на основе карты расстояний до источника засветки"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        overexposure_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
        reversed_mask = cv2.bitwise_not(overexposure_mask)
        dist = cv2.distanceTransform(reversed_mask,
                                     cv2.DIST_L2,
                                     cv2.DIST_MASK_3,
                                     reversed_mask)
        cv2.normalize(dist, dist, 0.0, radius, cv2.NORM_MINMAX)

        reversed_map_mask = np.zeros(dist.shape, dtype=np.uint8)
        reversed_map_mask[np.where(dist > 1)] = 255

        map_mask = np.zeros(dist.shape, dtype=np.uint8)
        map_mask[np.where(reversed_map_mask == 0)] = 255

        return map_mask

    @staticmethod
    def reduce(img):
        res = img.copy()
        for radius in range(20, 21):
            overexposed_mask = LightsStruckReducer.get_map_mask(res, radius)
            mask_inv = cv2.bitwise_not(overexposed_mask)
            source = cv2.bitwise_and(res, res, mask=mask_inv)
            res = cv2.bitwise_and(res, res, mask=overexposed_mask)
            res = LightsStruckReducer.gamma_correction(res, 1.06)
            res = cv2.add(source, res)
        return res


if __name__ == '__main__':
    video_name = 'klt.427.003.mp4'
    cap = cv2.VideoCapture('../data/processing/' + video_name)
    while cap.isOpened():
        succeed, frame = cap.read()
        if succeed:
            frame = LightsStruckReducer.reduce(frame)
            cv2.imshow(video_name, frame)
        else:
            cv2.destroyAllWindows()
            cap.release()
        cv2.waitKey(1)
