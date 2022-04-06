import cv2
import numpy as np
from matplotlib import pyplot as plt


class CarLightsStruckReducer:
    local_gamma = 1
    """Класс для уменьшения засветки от фар встречного ТС"""

    @staticmethod
    def reduce(img: np.ndarray) -> np.ndarray:
        res = img.copy()
        for radius in range(5, 1, -1):
            binary = CarLightsStruckReducer.get_overexposed_mask(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 220)
            not_bin = cv2.bitwise_not(binary)
            dist = cv2.distanceTransform(not_bin, cv2.DIST_L2, cv2.DIST_MASK_3, not_bin)
            cv2.normalize(dist, dist, float(radius), 0.0, cv2.NORM_MINMAX)
            not_mask = np.zeros(res.shape, dtype=np.uint8)
            not_mask[np.where(dist > 1)] = 255
            mask = np.zeros(res.shape, dtype=np.uint8)
            mask[np.where(not_mask == 0)] = 255
            m1,m2,m3 = cv2.split(mask)
            masked = CarLightsStruckReducer.get_masked(res, m1)
            reduced_overexposure = CarLightsStruckReducer.__gamma_correction(masked, 1.05)
            res = CarLightsStruckReducer.apply_masked_changes(res, reduced_overexposure, m1)
        return res

    @staticmethod
    def get_overexposed_mask(gray_img: np.ndarray, threshold: int, need_blur: bool = False):
        """
        Получить маску засвеченных областей с заданным порогом
        на основе перехода к grayscale
        """
        gray = gray_img.copy()
        if need_blur:
            gray = cv2.GaussianBlur(gray_img, (11, 11), 0)
        return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

    @staticmethod
    def get_masked(img: np.ndarray, mask: np.ndarray):
        """Получить фрагменты изображения под маской"""
        return cv2.bitwise_and(img, img, mask=mask)

    @staticmethod
    def __gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
        look_up_table = np.empty((1,256), np.uint8)
        for i in range(256):
            look_up_table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(img, look_up_table)
        return res

    @staticmethod
    def apply_masked_changes(img: np.ndarray,
                             changed_image: np.ndarray,
                             mask: np.ndarray):
        """Заменить в img маскированные области на области из changed_image"""
        img[np.where(mask == 255)] = changed_image[np.where(mask == 255)]
        return img

    @staticmethod
    def blurred_reduce(img: np.ndarray) -> np.ndarray:
        res = img.copy()
        binary = CarLightsStruckReducer.get_overexposed_mask(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 220)
        not_bin = cv2.bitwise_not(binary)
        dist = cv2.distanceTransform(not_bin, cv2.DIST_L2, cv2.DIST_MASK_3, not_bin)
        cv2.normalize(dist, dist, 4.0, 0.0, cv2.NORM_MINMAX)
        not_mask = np.zeros(res.shape, dtype=np.uint8)
        not_mask[np.where(dist > 1)] = 255
        mask = np.zeros(res.shape, dtype=np.uint8)
        mask[np.where(not_mask == 0)] = 255
        m1, m2, m3 = cv2.split(mask)
        masked = CarLightsStruckReducer.get_masked(res, m1)
        reduced_overexposure = CarLightsStruckReducer.__gamma_correction(masked, 2)
        reduced_overexposure = cv2.GaussianBlur(reduced_overexposure, (11, 11), 0)
        subtract = cv2.subtract(img, reduced_overexposure)
        blended = cv2.addWeighted(img, 1, subtract, 1, 0)
        return blended

def gamma_correction(img: np.ndarray, gamma: float):
    look_up_table = np.empty((1,256), np.uint8)
    for i in range(256):
        look_up_table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, look_up_table)
    return res

def get_map_mask(img, radius):
    binary = CarLightsStruckReducer.get_overexposed_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 220)
    cv2.imshow("bim", binary)
    not_bin = cv2.bitwise_not(binary)
    cv2.imshow("not_bin", not_bin)
    dist = cv2.distanceTransform(not_bin, cv2.DIST_L2, cv2.DIST_MASK_3, not_bin)
    cv2.normalize(dist, dist, 0.0, radius, cv2.NORM_MINMAX)
    not_mask = np.zeros(dist.shape, dtype=np.uint8)
    not_mask[np.where(dist > 1)] = 255
    cv2.imshow("not_mask", not_mask)
    mask = np.zeros(dist.shape, dtype=np.uint8)
    mask[np.where(not_mask == 0)] = 255
    cv2.imshow("mask", mask)
    cv2.waitKey()
    # cv2.imshow("map", mask)
    return mask

def reduce(img):
    res = img.copy()
    for radius in range(15, 20, 10):
        overexposed_mask = get_map_mask(res, radius)
        mask_inv = cv2.bitwise_not(overexposed_mask)
        source = cv2.bitwise_and(res, res, mask = mask_inv)
        res = cv2.bitwise_and(res, res, mask = overexposed_mask)
        res = gamma_correction(res, 1.06)
        res = cv2.add(source, res)
    # cv2.imshow("res", res)
    return res


if __name__ == '__main__':
    video_name = 'klt.427.003.mp4'
    cap = cv2.VideoCapture('../data/processing/' + video_name)
    while cap.isOpened():
        succeed, frame = cap.read()
        if succeed:
            frame = reduce(frame)
            cv2.imshow(video_name, frame)
        else:
            cv2.destroyAllWindows()
            cap.release()
        cv2.waitKey(1)
