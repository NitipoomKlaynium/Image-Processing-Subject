import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from keras.models import Model, Input
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import glob

face_dataset = "face_dataset/lfw/**/*.jpg"
face_mini = "face_mini/**/*.jpg"
filenames = glob.glob(face_mini)

all_images = []

for fname in filenames[:500] :
    img = image.load_img(fname, target_size=(100, 100), interpolation="nearest")
    img = image.img_to_array(img)
    img /= 255
    all_images.append(img)
all_images = np.array(all_images)

train_x, test_x = train_test_split(all_images, random_state=32, test_size=0.3)
train_x, val_x = train_test_split(train_x, random_state=32, test_size=0.3)

amount = 5;

for i in range(amount) :
    plt.subplot(2, amount, i + 1)
    plt.axis('off')
    plt.imshow(train_x[i])
    
    plt.subplot(2, amount, i + 1 + amount)
    plt.axis('off')
    plt.imshow(test_x[i])
    
plt.show()