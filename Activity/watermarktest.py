from unittest import result
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
import glob
import random

watermark = cv.imread('D:/work/Stuff/watermark/watermark-1.png', cv.IMREAD_UNCHANGED)
opacity = 0.4
watermark = np.uint8(watermark[:, :] * [1, 1, 1, opacity])

files = sorted(glob.glob('D:/work/Stuff/Facebook1/*.jpg'))

count = 0

for file in files :
    img = cv.cvtColor(cv.imread(file), cv.COLOR_BGR2BGRA)

    img_h = img.shape[0]
    img_w = img.shape[1]

    if img.shape[0] > img.shape[1] :
        cur_watermark = cv.resize(watermark, (img_h, img_h))
    else :
        cur_watermark = cv.resize(watermark, (img_w, img_w))
    cur_watermark = cur_watermark[0:img_h, 0:img_w]

    plt.subplot(1, 2, 2),plt.imshow(img)
    plt.axis('off')
    plt.title('Image')

    x = img.copy()

    img[:, :, 0] = cv.add(cur_watermark[:, :, 0] * (cur_watermark[:, :, 3] / 255) , img[:, :, 0] * ((img[:, :, 3] - cur_watermark[:, :, 3]) / 255))
    img[:, :, 1] = cv.add(cur_watermark[:, :, 1] * (cur_watermark[:, :, 3] / 255) , img[:, :, 1] * ((img[:, :, 3] - cur_watermark[:, :, 3]) / 255))
    img[:, :, 2] = cv.add(cur_watermark[:, :, 2] * (cur_watermark[:, :, 3] / 255) , img[:, :, 2] * ((img[:, :, 3] - cur_watermark[:, :, 3]) / 255))

    cv.imwrite("D:/work/Stuff/image_data_set_1/Watermark 2/image_" + str(count).zfill(5) + ".jpg", img)
    count += 1
    
    # plt.subplot(1, 2, 1),plt.imshow(img)
    # plt.axis('off')
    # plt.title('Image')

    # plt.show()