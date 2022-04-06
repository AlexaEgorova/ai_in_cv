import cv2
import numpy as np

def gamma_correction(img: np.ndarray, gamma: float):
    look_up_table = np.empty((1,256), np.uint8)
    for i in range(256):
        look_up_table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, look_up_table)
    return res


def get_overexposed_mask(gray_img: np.ndarray, threshold: int, need_blur: bool = True):
    """
    Получить маску засвеченных областей с заданным порогом
    на основе перехода к grayscale
    """
    gray = gray_img.copy()
    if need_blur:
        gray = cv2.GaussianBlur(gray_img, (11, 11), 0)
    return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]


def get_masked(img: np.ndarray, mask: np.ndarray):
    """Получить фрагменты изображения под маской"""
    return cv2.bitwise_and(img, img, mask=mask)


def apply_masked_changes(img: np.ndarray,
                         changed_image: np.ndarray,
                         mask: np.ndarray):
    """Заменить в img маскированные области на области из changed_image"""
    img[np.where(mask == 255)] = changed_image[np.where(mask == 255)]
    return img

def get_map_mask(img, radius):
    bin = get_overexposed_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 220)
    not_bin = cv2.bitwise_not(bin)
    dist = cv2.distanceTransform(not_bin, cv2.DIST_L2, cv2.DIST_MASK_3, not_bin)
    cv2.normalize(dist, dist, 0.0, radius, cv2.NORM_MINMAX)
    not_mask = np.zeros(dist.shape, dtype=np.uint8)
    not_mask[np.where(dist > 1)] = 255
    mask = np.zeros(dist.shape, dtype=np.uint8)
    mask[np.where(not_mask == 0)] = 255
    # cv2.imshow("map", mask)
    return mask

def reduce(img):
    source = img.copy()
    res = img.copy()
    for radius in range(10, 15):
        overexposed_mask = get_map_mask(res, radius)
        mask_inv = cv2.bitwise_not(overexposed_mask)
        source = cv2.bitwise_and(res, res, mask = mask_inv)
        res = cv2.bitwise_and(res, res, mask = overexposed_mask)
        res = gamma_correction(res, 1.1)
        res = cv2.add(source, res)
    cv2.imshow("res", res)
    # return img

if __name__ == '__main__':
    img = cv2.imread('test_1.png')
    img = cv2.resize(img, (650, 350), interpolation=cv2.INTER_AREA)
    reduce(img)
    # get_map_mask(img)
    # cv2.imshow('img', img)
    cv2.waitKey()