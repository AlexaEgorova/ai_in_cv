#Уменьшить засветку от света фар встречного автомобиля [klt.427.003]

import numpy as np
import cv2


def gamma_correction(img: np.array, gamma: float):
    look_up_table = np.empty((1,256), np.uint8)
    for i in range(256):
        look_up_table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, look_up_table)
    return res


def get_overexposed_mask(gray_img: np.array, threshold: int, need_blur: bool = True):
    """
    Получить маску засвеченных областей с заданным порогом
    на основе перехода к grayscale
    """
    gray = gray_img.copy()
    if need_blur:
        gray = cv2.GaussianBlur(gray_img, (11, 11), 0)
    return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]


def get_overexposed_mask_by_channels(img: np.array, threshold: int, need_blur: bool = True):
    """
        Получить маску засвеченных областей с заданным порогом
        на основе разложения на каналы
    """
    b, g, r = cv2.split(img)
    masks = [get_overexposed_mask(channel, threshold, need_blur) for channel in [b, g, r]]
    return sum(masks)


def get_masked(img: np.array, mask: np.array):
    """Получить фрагменты изображения под маской"""
    return cv2.bitwise_and(img, img, mask=mask)


def apply_masked_changes(img: np.array, changed_image: np.array, mask: np.array):
    """Заменить в img маскированные области на области из changed_image"""
    img[np.where(mask == 255)] = changed_image[np.where(mask == 255)]
    return img


def reduce_light_struck(img: np.array):
    cv2.imshow("img", img)
    img = gamma_correction(img, 1.1)
    res = img.copy()
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    for th in range(250, 50, -10):
        overexposed_mask = get_overexposed_mask(gray, th, need_blur=True)
        overexposed = get_masked(res, overexposed_mask)
        reduced_overexposure = gamma_correction(overexposed, 1.025)
        res = apply_masked_changes(res, reduced_overexposure, overexposed_mask)

    cv2.imshow('res', res)


if __name__ == '__main__':
    img = cv2.imread('test_1.png')
    img = cv2.resize(img, (650, 350), interpolation=cv2.INTER_AREA)

    reduce_light_struck(img)

    cv2.waitKey(0)