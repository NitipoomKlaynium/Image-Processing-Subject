import cv2 as cv
import numpy as np
from matplotlib import  pyplot as plt
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from skimage import io
img_2 = img_to_array(cv.resize(io.imread("Swing_224x224.png"),(224,224))) 
# cv.cvtColor(myImg, cv.COLOR_BGR2RGB)
img_2 = img_2.astype("float")


img_mean = [103.939, 116.779,123.68]

for h in range(0,224) :
  for w in range(0,224) :
    for c in range(0,3) :
      img_2[h][w][c] = img_2[h][w][c] - img_mean[c]

img_2 = expand_dims(img_2,axis=0)
print(img_2.shape)
plt.subplot(1,1,1),
plt.imshow(img_2[0,: :, :,],cmap='jet')
plt.show()