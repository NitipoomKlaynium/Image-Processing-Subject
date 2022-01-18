import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_hwc = cv.imread('Swing.png')

img_chw = np.moveaxis(img_hwc, -1, 0)

print(img_hwc.shape)
print("============================")
print(img_chw.shape)

