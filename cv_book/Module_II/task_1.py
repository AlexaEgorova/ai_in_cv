#Уменьшить засветку от света фар встречного автомобиля [klt.427.003]

import numpy as np

import cv2

def reduce_car_backlights(img: np.array):
    gamma = 0.8
    look_up_table = np.empty((1,256), np.uint8)
    for i in range(256):
        look_up_table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, look_up_table)

    return res


if __name__ == '__main__':
    img = cv2.imread('test_1.png')
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    img_remastered = reduce_car_backlights(img)

    cv2.imshow('lights', img)
    cv2.imshow('no_lights', img_remastered)

    cv2.waitKey(0)