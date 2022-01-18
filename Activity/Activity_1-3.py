import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

img_8 = cv.imread('Swing.png', cv.IMREAD_GRAYSCALE)

img_h = img_8.shape[0]
img_w = img_8.shape[1]

img_4 = np.asarray([[0]*img_w]*img_h, np.uint8)

bit_depth = 4;
Qlevel = 2**bit_depth;

for h in range(img_h) :
    for w in range(img_w) :
        img_4[h][w] = math.floor(((float(img_8[h][w]))/(float(255))) * Qlevel)
        
plt.subplot(121),plt.imshow(img_8, cmap = 'gray')
plt.title('8')
plt.subplot(122),plt.imshow(img_4, cmap = 'gray')
plt.title('4')

plt.tight_layout()
plt.show()
