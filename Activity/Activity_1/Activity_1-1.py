import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Swing.png')
        
(B, G, R) = cv.split(img)

plt.subplot(241),plt.imshow(img)
plt.title('BGR')
plt.subplot(242),plt.imshow(B, cmap = 'gray')
plt.title('B')
plt.subplot(243),plt.imshow(G, cmap = 'gray')
plt.title('G')
plt.subplot(244),plt.imshow(R, cmap = 'gray')
plt.title('R')

img_h = img.shape[0]
img_w = img.shape[1]

for h in range(img_h) :
    for w in range(img_w) :
        img[h][w] = img[h][w][::-1]

plt.subplot(245),plt.imshow(img)
plt.title('RGB')
plt.subplot(246),plt.imshow(R, cmap = 'gray')
plt.title('R')
plt.subplot(247),plt.imshow(G, cmap = 'gray')
plt.title('G')
plt.subplot(248),plt.imshow(B, cmap = 'gray')
plt.title('B')

plt.tight_layout()
plt.show()
