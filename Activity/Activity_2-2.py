import cv2 as cv
import numpy as np
from matplotlib import image, pyplot as plt

def crop(image_src: np.ndarray, size: set, origin = (0, 0)) :
    result_h = size[0]
    result_w = size[1]
    img_result = np.asarray(result_h * [result_w * [[0, 0, 0]]], np.uint8)
    for h in range(0, result_h) :
        for w in range(0, result_w) :
            img_result[h][w] = image_src[origin[0] + h][origin[1] + w]
    return img_result

def zoom(image_src, multiplier: float) :
    width = int(image_src.shape[1] * multiplier)
    height = int(image_src.shape[0] * multiplier)
    dim = (width, height)
    image_result = cv.resize(image_src, dim)
    return image_result

if __name__ == "__main__" :
    
    # Resize image
    img_swing = cv.imread('Swing.png')
    img_swing = crop(img_swing, (600, 600), (0, 50))

    img_nature = cv.imread('Nature.jpg')
    img_nature = zoom(img_nature, 2)
    img_nature = crop(img_nature, (600, 600), (0, 100))
    
    # Generate weight array
    weight_array = [(1 - float(i)/30, float(i)/30) for i in range(31)]
    # x = (list)(a + b for a, b in weight_array)
    # print(x)
    
    # Create vdโอ writer
    vdโอ_fourcc = cv.VideoWriter_fourcc(*'MP4V')
    vdโอ = cv.VideoWriter("Swing_Nature.mp4", vdโอ_fourcc, 10.0, (600,600))
    
    # Frame writing
    for i in range(20) :
        vdโอ.write(img_swing)
    
    for i in range(len(weight_array)) :
        frame = np.uint8(img_swing * weight_array[i][0] + img_nature * weight_array[i][1])
        vdโอ.write(frame)
        
    for i in range(20) :
        vdโอ.write(img_nature)
        
    for i in range(len(weight_array)) :
        frame = np.uint8(img_nature * weight_array[i][0] + img_swing * weight_array[i][1])
        vdโอ.write(frame)
    
    for i in range(20) :
        vdโอ.write(img_swing)
    
    # vdโอ Release
    vdโอ.release()
 