#Уменьшить засветку от света фар встречного автомобиля [klt.427.003]

import numpy as np

import cv2

def increase_sharpness(img: np.array):
    return img


if __name__ == '__main__':
    img = cv2.imread('test_5.png')
    img = increase_sharpness(img)