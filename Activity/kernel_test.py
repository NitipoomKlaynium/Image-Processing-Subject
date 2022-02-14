from operator import mod
import cv2 as cv
import numpy as np
from matplotlib import  pyplot as plt

img = cv.imread('test_image_kernel.png', cv.IMREAD_GRAYSCALE);

img_h = img.shape[0]
img_w = img.shape[1]

img_out = np.asarray(img_h * [img_w * [0]], np.uint8)

kernel = [[ 0, -1,  0],
          [-1,  5, -1],
          [ 0, -1,  0]]

img_for_process = np.asarray((img_h + 1) * [(img_w + 1) * [0]], np.uint8)
 
for h in range(img_for_process.shape[0]) :
    for w in range(img_for_process.shape[1]) : 
        try :
            img_for_process[h][w] = img[h][w]
        except :
            img_for_process[h][w] = 0
print(img_for_process)
        

for h in range(img_h) :
    for w in range(img_w) :
        sum = 0;
        for kernel_h in range(3) :
            for kernel_w in range(3) :
                sum += img_for_process[h + kernel_h - 1][w + kernel_w - 1] * kernel[kernel_h][kernel_w]
        if sum < 0 :
            sum = 0
        elif sum > 255 :
            sum = 255
        img_out[h][w] = sum


plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(122)
plt.imshow(img_out, cmap='gray')
plt.title('Output')

plt.show()