import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
from keras.preprocessing import image
from sklearn.cluster import KMeans
from skimage.feature import _hog
from skimage import exposure
from skimage.measure import label
import random

if __name__ == "__main__" :
    files = glob.glob("Animals/**/*.jpg") + glob.glob("Animals/**/*.jpeg") + glob.glob("Animals/**/*.webp")
    for file in files :
        print(cv.imread(file).shape)
    # img = cv.imread(random.choice(files))
    # blur = cv.GaussianBlur(img, (5, 5), 0)
    
    
    # plt.subplot(2, 4, 1)
    # plt.imshow(cv.cvtColor(blur, cv.COLOR_BGR2RGB))
    # plt.title("Input")
    
    # plt.subplot(2, 4, 2)
    # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # plt.title("Histogram of Oriented Gradients")
    
    plt.show()