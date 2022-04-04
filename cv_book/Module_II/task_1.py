import cv2
import numpy as np
from matplotlib import pyplot as plt


class CarLightsStruckReducer:
    local_gamma = 1
    """Класс для уменьшения засветки от фар встречного ТС"""
    @staticmethod
    def reduce_light_struck(img: np.ndarray, need_update_gamma: bool = False) -> np.ndarray:
        if need_update_gamma:
            CarLightsStruckReducer.local_gamma = CarLightsStruckReducer.__calculate_gamma(img)
        res = CarLightsStruckReducer.__gamma_correction(img, CarLightsStruckReducer.local_gamma)
        return res

    @staticmethod
    def __calculate_gamma(img):
        '''Поиск порога распредлений, до которого лежит 85% всех значений пикселей'''
        hist = plt.hist(img.ravel(), 256, [0, 256])
        threshold = CarLightsStruckReducer.__bin_search(hist[0])
        dark_sum = sum(hist[0][:threshold])
        bright_sum = sum(hist[0][threshold:])
        diff = abs(dark_sum - bright_sum)
        accumulate = dark_sum + bright_sum
        if dark_sum > bright_sum:
            gamma = accumulate / diff
        else:
            gamma = diff / accumulate
        return gamma

    @staticmethod
    def __bin_search(iterable):
        '''Бинпоиск порога по гистограмме распределений'''
        l, r = 0, 256
        threshold = 0.85 * sum(iterable)
        while r - l > 1:
            mid = (l + r) // 2
            if sum(iterable[:mid]) > threshold:
                r = mid
            else:
                l = mid
        return l

    @staticmethod
    def __get_percentile_index(hist_values: np.ndarray) -> float:
        # может заменить бинсёрч, но работает плёха
        threshold_value = np.percentile(hist_values, 50, interpolation='higher')
        threshold = abs(hist_values - threshold_value).argmin()
        return threshold

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


if __name__ == '__main__':
    video_name = 'klt.427.003.mp4'
    cap = cv2.VideoCapture('../data/processing/' + video_name)
    frame_idx = 0
    while cap.isOpened():
        succeed, frame = cap.read()
        frame_idx += 1
        if succeed:
            frame = CarLightsStruckReducer.reduce_light_struck(frame, frame_idx % 50 == 1)
            cv2.imshow(video_name, frame)
        else:
            cv2.destroyAllWindows()
            cap.release()
        if frame_idx % 100 == 1:
            cv2.waitKey(1)
        else:
            cv2.waitKey(10)
