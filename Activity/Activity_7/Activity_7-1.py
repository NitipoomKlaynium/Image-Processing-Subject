import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == "__main__" :
    img = cv.cvtColor(cv.imread("Swing.png"), cv.COLOR_BGR2RGB)
    
    Reduce_factors = [2, 7, 15]
    inter_methods = [cv.INTER_NEAREST,cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA]
    inter_methods_str = ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_AREA"]
    for i in range(len(Reduce_factors)) :
        for j in range(len(inter_methods)) :
            Scale_factors = 1 / Reduce_factors[i]
            dim = (int(img.shape[0] * Scale_factors), int(img.shape[1] * Scale_factors))
            img_effect = cv.resize(src=img, dsize=dim, interpolation=inter_methods[j])
            plt.subplot(len(Reduce_factors), len(inter_methods), j + (i * len(inter_methods)) + 1)
            # plt.axis('off')
            plt.imshow(img_effect)
            plt.title(inter_methods_str[j] + " (" + str(Reduce_factors[i]) + ")")
    
    plt.show()