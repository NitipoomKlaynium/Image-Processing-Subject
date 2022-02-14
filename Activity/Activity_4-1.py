from operator import mod
import cv2 as cv
import numpy as np
from matplotlib import  pyplot as plt
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

img = cv.cvtColor(cv.imread('atk-1.png'), cv.COLOR_BGR2RGB);

img = img_to_array(img)
# img = np.array(img, np.float32)
# print(img)
# print(type(img[0][0][0]))

img = np.expand_dims(img, axis=0)

img_ready = preprocess_input(img)

# print(img_ready)

model = Model(inputs=model.inputs, outputs=model.layers[1].output)

# print(model)

feature_maps = model.predict(img_ready)

# print(feature_maps.ndim)
# print(len(feature_maps))
# print(len(feature_maps[0]))
# print(len(feature_maps[0][0]))
# print(len(feature_maps[0][0][0]))
# print(len(feature_maps[0][0][0][0]))

# print(feature_maps[0, :, :, 31])

square = 8

for i in range(square):
	for j in range(square):
		# specify subplot and turn of axis
		pos = (i * 8) + j
		ax = plt.subplot(square, square, pos + 1)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(feature_maps[0, :, :, pos], cmap='jet')
# show the figure
plt.show()