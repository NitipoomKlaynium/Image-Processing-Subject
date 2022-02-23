import cv2 as cv
from cv2 import sort
import numpy as np
import glob, os, time, datetime

def crop(image_src: np.ndarray, size: set, origin = (0, 0)) :
    img_result = image_src[origin[0]:size[0], origin[1]:size[1]]
    return img_result

def zoom(image_src, multiplier: float) :
    width = int(image_src.shape[1] * multiplier)
    height = int(image_src.shape[0] * multiplier)
    dim = (width, height)
    image_result = cv.resize(image_src, dim)
    return image_result

load_path = "D:/work/Stuff/Facebook1/"

filenames = glob.glob(load_path + "**")
filenames = sorted(filenames, key = lambda file: datetime.datetime.fromtimestamp(os.stat(file).st_ctime))
filenames = filenames[::-1]

save_path = "D:/work/Stuff/image_data_set/"
start = 0

for filename in filenames :
    img_read = cv.imread(filename)
    h = img_read.shape[0]
    w = img_read.shape[1]
    if h > w :
        next_h = int((h - w) / 2)
        img_crop = img_read[next_h:(next_h + w), 0:w]
    else :
        next_w = int((w - h) / 2)
        img_crop =img_read[0:h, next_w:(next_w + h)]
    img_resize = cv.resize(img_crop, (150, 150))
    
    cv.imwrite(save_path + "img_" + "{:05d}".format(start) + ".jpg" , img_resize)
    start += 1
    