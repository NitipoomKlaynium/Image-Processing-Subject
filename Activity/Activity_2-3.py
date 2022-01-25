import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_BGR = cv.imread('Swing.png')

img = cv.cvtColor(IMG_BGR, cv.COLOR_BGR2RGB)  

img_h = img.shape[0]
img_w = img.shape[1]

img_mask = np.asarray(img_h * [img_w * [(0, 0, 0)]], np.uint8)

for h in range(300, 600) :
    for w in range(300, 660) :
        img_mask[h][w] = (255, 255, 255)

img_obj_area = cv.bitwise_and(img, img_mask)

plt.subplot(1, 3, 1)
plt.title('Original'), plt.imshow(img)

plt.subplot(1, 3, 2)
plt.title('Image Mask'), plt.imshow(img_mask)

plt.subplot(1, 3, 3)
plt.title('Bitwise_AND() result'), plt.imshow(img_obj_area)

plt.tight_layout()
plt.show()
