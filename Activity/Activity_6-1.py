import cv2 as cv
import numpy as np
from keras.models import Model, Input
from keras.preprocessing import image
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
import glob
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping

img_data_set = "D:/work/Stuff/image_data_set/*.jpg"
filenames = glob.glob(img_data_set)

all_images = []
H, W = 150, 150

for fname in filenames[2000:2500] :
    img = image.load_img(fname, target_size=(H, W), interpolation="nearest")
    img = image.img_to_array(img)
    img /= 255
    all_images.append(img)
all_images = np.array(all_images)

train_x, test_x = train_test_split(all_images, random_state=32, test_size=0.3)
train_x, val_x = train_test_split(train_x, random_state=32, test_size=0.3)

noise_factor = 0.4
Nmean = 0
Nstd = 1

x_train_noisy = train_x + (noise_factor * np.random.normal(loc=Nmean, scale=Nstd, size=train_x.shape) )
x_val_noisy = val_x + ( noise_factor * np.random.normal(loc=Nmean, scale=Nstd, size=val_x.shape) )
x_test_noisy = test_x + ( noise_factor * np.random.normal(loc=Nmean, scale=Nstd, size=test_x.shape) )

amount = 5;

for i in range(amount) :
    plt.subplot(2, amount, i + 1)
    plt.axis('off')
    plt.imshow(x_train_noisy[i])
    plt.title("Add Noise")
    
    plt.subplot(2, amount, i + 1 + amount)
    plt.axis('off')
    plt.imshow(train_x[i])
    plt.title("Original")
 
plt.show()