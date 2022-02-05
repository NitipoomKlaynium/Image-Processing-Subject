import cv2 as cv
import numpy as np
from matplotlib import image, pyplot as plt

if __name__ == "__main__" :
    
    color = ('r','g','b')
    
    img = cv.cvtColor(cv.imread('canyon.png'), cv.COLOR_BGR2RGB);
    
    # Normal
    
    plt.subplot(221)
    plt.imshow(img)
    plt.title('Original Color Image')


    plt.subplot(222)
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0,256])
        # print(np.uint(histr))
        # print("=====")
        print(type(histr))
        plt.plot(histr, color = col)
        plt.xlim([0,256])
    
    plt.title('Hist Original Image')
       
    # #Equalize Hist
    
    (R,G,B) = cv.split(img)
    R_hist = cv.equalizeHist(R)
    G_hist = cv.equalizeHist(G)
    B_hist = cv.equalizeHist(B)
    
    # print(img[..., 0])
    
    # img_eql_hst = cv.merge([R_hist,G_hist,B_hist])
    
    # plt.subplot(223)
    # plt.imshow(img_eql_hst)
    # plt.title('Equalized Image')
    
    # plt.subplot(224)
    # for i,col in enumerate(color):
    #     histr = cv.calcHist([img_eql_hst],[i],None,[256],[0,256])
    #     plt.plot(histr,color = col)
    #     plt.xlim([0,256])
    # plt.title('Hist Equalized Image')

    plt.tight_layout()
    plt.show()
