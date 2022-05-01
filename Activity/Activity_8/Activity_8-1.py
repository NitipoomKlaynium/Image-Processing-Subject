import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
from keras.preprocessing import image
from scipy import spatial
import random

if __name__ == "__main__" :
    files = glob.glob("D:/work/Stuff/image_data_set_1/Original/*.jpg")
    main_image = cv.imread(random.choice(files))
    main_image = main_image[0:(main_image.shape[0] // 40) * 40, 0:(main_image.shape[1] // 40) * 40]
    tile_size = np.array([40, 40])
    
    
    
    main_image_feature = np.empty((int(main_image.shape[0] / 40) , int(main_image.shape[1] / 40), 3), dtype=np.uint8)
    
    # print(main_image.shape)
    # print(main_image_feature.shape)
    
    for i in range(main_image_feature.shape[0]) :
        for j in range(main_image_feature.shape[1]) :
            main_image_feature[i][j] = main_image[i*tile_size[0] : (i+1)*tile_size[0],
                                                j*tile_size[1]: (j+1)*tile_size[1]].mean(axis=(0,1))

    
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(main_image, cv.COLOR_BGR2RGB))
    plt.title("Original")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(main_image_feature, cv.COLOR_BGR2RGB))
    plt.title("Main image feature")
 
    plt.show()
    
    