import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_BGR = cv.imread('Swing.png')

# RGB
IMG_RGB = cv.cvtColor(IMG_BGR, cv.COLOR_BGR2RGB)  
(IMG_RGB_R, IMG_RGB_G, IMG_RGB_B) = cv.split(IMG_RGB)

plt.subplot(4, 4, 1)
plt.title('RGB'), plt.imshow(IMG_RGB)

plt.subplot(4, 4, 2)
plt.title('R'), plt.imshow(IMG_RGB_R, cmap = 'gray')

plt.subplot(4, 4, 3)
plt.title('G'), plt.imshow(IMG_RGB_G, cmap = 'gray')

plt.subplot(4, 4, 4)
plt.title('B'), plt.imshow(IMG_RGB_B, cmap = 'gray')

# HSV
IMG_HSV = cv.cvtColor(IMG_RGB, cv.COLOR_RGB2HSV)  
(IMG_HSV_H, IMG_HSV_S, IMG_HSV_V) = cv.split(IMG_HSV)

plt.subplot(4, 4, 5)
plt.title('HSV'), plt.imshow(IMG_HSV)

plt.subplot(4, 4, 6)
plt.title('H'), plt.imshow(IMG_HSV_H, cmap = 'gray')

plt.subplot(4, 4, 7)
plt.title('S'), plt.imshow(IMG_HSV_S, cmap = 'gray')

plt.subplot(4, 4, 8)
plt.title('V'), plt.imshow(IMG_HSV_V, cmap = 'gray')

# HLS
IMG_HLS = cv.cvtColor(IMG_RGB, cv.COLOR_RGB2HLS)  
(IMG_HLS_H, IMG_HLS_L, IMG_HLS_S) = cv.split(IMG_HLS)

plt.subplot(4, 4, 9)
plt.title('HLS'), plt.imshow(IMG_HLS)

plt.subplot(4, 4, 10)
plt.title('H'), plt.imshow(IMG_HLS_H, cmap = 'gray')

plt.subplot(4, 4, 11)
plt.title('L'), plt.imshow(IMG_HLS_L, cmap = 'gray')

plt.subplot(4, 4, 12)
plt.title('S'), plt.imshow(IMG_HLS_S, cmap = 'gray')

# YCrCb
IMG_YCrCb = cv.cvtColor(IMG_RGB, cv.COLOR_RGB2YCR_CB)  
(IMG_YCrCb_Y, IMG_YCrCb_CR, IMG_YCrCb_CB) = cv.split(IMG_YCrCb)

plt.subplot(4, 4, 13)
plt.title('YCrCb'), plt.imshow(IMG_YCrCb)

plt.subplot(4, 4, 14)
plt.title('Y'), plt.imshow(IMG_YCrCb_Y, cmap = 'gray')

plt.subplot(4, 4, 15)
plt.title('Cr'), plt.imshow(IMG_YCrCb_CR, cmap = 'gray')

plt.subplot(4, 4, 16)
plt.title('Cb'), plt.imshow(IMG_YCrCb_CB, cmap = 'gray')

plt.tight_layout()
plt.show()