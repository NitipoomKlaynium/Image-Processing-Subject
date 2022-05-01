import cv2 as cv
import numpy as np
import math
import matplotlib.image as mpimg
from skimage import io
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from scipy import signal

model = VGG16()
# model.summary()

kernels, biases = model.layers[1].get_weights()
# print(model.layers[1].get_config())

img = cv.cvtColor(cv.imread('Swing_224x224.png'), cv.COLOR_BGR2RGB);

img = img_to_array(img)
# img = np.array(img, np.float32)
# print(img)
# print(type(img[0][0][0]))

img = np.expand_dims(img, axis=0)

img_2 = img_to_array(cv.resize(io.imread("Swing_224x244.png"),(224,224))) 
# cv.cvtColor(myImg, cv.COLOR_BGR2RGB)
img_2 = img_2.astype("float")


img_mean = [103.939, 116.779,123.68]

for h in range(0,224) :
  for w in range(0,224) :
    for c in range(0,3) :
      img_2[h][w][c] = img_2[h][w][c] - img_mean[c]

img_2 = expand_dims(img_2, axis=0)

img_result = np.copy(img[0,:,:])
print(img_result.shape)
for i in range(len(kernels[0,0,0,:])) :
  for j in range(3) :
    img_result[:,:,j] = signal.convolve2d(img_2[0,:,:,j],kernels[:,:,j,i],mode='same',boundary='fill',fillvalue=0)
  Image_sum = img_result[: , : , 0] + img_result[: , : , 1] + img_result[ : , : , 2]
  Image_sum = np.where(Image_sum<0,0,Image_sum)

  plt.subplot(8, 8, i+1)
  # plot filter channel in grayscale
  plt.imshow(Image_sum, cmap='jet')

# show the figure
plt.show()