#Уменьшить засветку от света фар встречного автомобиля [klt.427.003]

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


def get_overexposed_mask_by_channels(img: np.ndarray, threshold: int, need_blur: bool = True):
    """
        Получить маску засвеченных областей с заданным порогом
        на основе разложения на каналы
    """
    b, g, r = cv2.split(img)
    masks = [get_overexposed_mask(channel, threshold, need_blur) for channel in [b, g, r]]
    return sum(masks)


def get_masked(img: np.ndarray, mask: np.ndarray):
    """Получить фрагменты изображения под маской"""
    return cv2.bitwise_and(img, img, mask=mask)


def apply_masked_changes(img: np.ndarray,
                         changed_image: np.ndarray,
                         mask: np.ndarray):
    """Заменить в img маскированные области на области из changed_image"""
    img[np.where(mask == 255)] = changed_image[np.where(mask == 255)]
    return img


def reduce_light_struck(img: np.ndarray):
    cv2.imshow("img", img)
    img = gamma_correction(img, 1.1)
    res = img.copy()
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    for th in range(250, 50, -10):
        overexposed_mask = get_overexposed_mask(gray, th, need_blur=True)
        overexposed = get_masked(res, overexposed_mask)
        reduced_overexposure = gamma_correction(overexposed, 1.025)
        res = apply_masked_changes(res, reduced_overexposure, overexposed_mask)
    return res
    # cv2.imshow('accumulative gamma correction', res)


def get_overexposure_map(img: np.ndarray):
    binary = get_overexposed_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 220)     #бинаризацией нашла фары
    not_binary = cv2.bitwise_not(binary)                                                 #перевернула
    dist = cv2.distanceTransform(not_binary, cv2.DIST_L2, cv2.DIST_MASK_3, not_binary)       #сделала карту расстояний от светлый учатсков до темных
    cv2.normalize(dist, dist, -30.0, 0.0, cv2.NORM_MINMAX)    #нормализацией регулирую черноту и область
    return abs(dist)

def apply_overexposure_map_by_channgels(img: np.ndarray, map: np.ndarray):
    b, g, r = cv2.split(img)
    res_b = np.uint8(abs(b - map))
    res_g = np.uint8(abs(g - map))
    res_r = np.uint8(abs(r - map))
    res = cv2.merge((res_b, res_g, res_r))
    return res


def Yana_do_smth(img: np.ndarray):
    bin = get_overexposed_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 220)     #бинаризацией нашла фары
    not_bin = cv2.bitwise_not(bin)                                                 #перевернула
    dist = cv2.distanceTransform(not_bin, cv2.DIST_L2, cv2.DIST_MASK_3, not_bin)       #сделала карту расстояний от светлый учатсков до темных
    cv2.normalize(dist, dist, 0.0, 5.0, cv2.NORM_MINMAX)                               #нормализацией регулирую черноту и область
    rgb_mask = cv2.cvtColor(dist, cv2.COLOR_GRAY2BGR)                             #делаю чб трехканальным, диапазон от 0 до 1 !!!!!! не знаю что с этим делать
    res_img = np.uint8(0.6 * img + 0.4 * rgb_mask)                                   #складываю с пропорциями 60 на 40
    b, g, r = cv2.split(res_img)
    r[r > 150] = 200
    red_img = cv2.merge([b, g, r])
    inv_mask = 1 - dist
    rgb_inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
    res_inv_img = np.uint8(img - rgb_inv_mask)
    b, g, r = cv2.split(img)
    b = b - inv_mask
    g = g - inv_mask
    r = r - inv_mask
    res_inv_rgb_res = cv2.merge([r, g, b])

    cv2.imshow('mask', rgb_mask)
    cv2.imshow('added proportionally with mask', res_img)
    cv2.imshow('increased red', red_img)
    cv2.imshow('mask.inv', rgb_inv_mask)
    cv2.imshow('substructed mask.inv', res_inv_img)
    cv2.imshow('substructed mask.inv by channels', res_inv_rgb_res)
    cv2.imshow('added gamma to mask.added', gamma_correction(res_img, 0.9))


def reduce_light_struck_bez_pontov(img: np.ndarray):
    res = gamma_correction(img, 1.4)
    # cv2.imshow('gamma correction 1.4', res)
    return res


if __name__ == '__main__':
    img = cv2.imread('test_1.png')
    img = cv2.resize(img, (650, 350), interpolation=cv2.INTER_AREA)
    # cv2.imshow('img', img)

    res = reduce_light_struck(img)
    cv2.imshow("adaptive", np.hstack((img, res)))
    reduce_light_struck(img)
    # reduce_light_struck_bez_pontov(img)
    # Yana_do_smth(img)
    cv2.waitKey()

    # cap = cv2.VideoCapture('../data/processing/klt.427.003.mp4')
    # count = 0
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if ret:
    #         map = get_overexposure_map(frame)
    #         # frame = apply_overexposure_map_by_channgels(frame, map)
    #         frame = reduce_light_struck_bez_pontov(frame)
    #         cv2.imshow('klt.427.003.mp4', frame)
    #         count = count + 1
    #     cv2.waitKey(5)
    # cap.release()
    # cv2.destroyAllWindows()  # destroy all opened windows
