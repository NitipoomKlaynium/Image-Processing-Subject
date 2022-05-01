import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

if __name__ == "__main__" :
    img = cv.imread("Swing.png")
    
    vdโอ_fourcc = cv.VideoWriter_fourcc(*'mp4v')
    vdโอ = cv.VideoWriter("Activity_7.mp4", vdโอ_fourcc, 2.0, (img.shape[1], img.shape[0]))
    
    fill_method = ['constant', 'nearest', 'reflect', 'wrap']
    
    Npic = 10
    rotation_range = 90
    width_shift_range = 1.0
    height_shift_range = 1.0
    shear_range = 1.0
    zoom_range = 0.5
    horizontal_flip = True
    vertical_flip = True
        
    for m in fill_method :
        datagen = ImageDataGenerator(rotation_range=rotation_range,
                                     width_shift_range=width_shift_range,
                                     height_shift_range=height_shift_range,
                                     shear_range=shear_range,
                                     zoom_range=zoom_range,
                                     horizontal_flip=horizontal_flip,
                                     vertical_flip=vertical_flip,
                                     fill_mode=m)
        
        im1 = expand_dims(img, axis=0)
        pic = datagen.flow(im1, batch_size = 1)
        
        for i in range(1, Npic) :
            batch = pic.next()
            im_result = batch[0].astype('uint8')
            vdโอ.write(im_result)

    vdโอ.release()