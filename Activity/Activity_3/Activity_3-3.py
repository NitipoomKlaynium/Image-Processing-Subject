import cv2 as cv
import numpy as np
from matplotlib import image, pyplot as plt
from skimage import exposure
from skimage.exposure import cumulative_distribution

def crop(image_src: np.ndarray, size: set, origin = (0, 0)) :
    img_result = image_src[origin[0]:(origin[0] + size[0]), origin[1]:(origin[1] + size[1])]
    return img_result

def zoom(image_src, multiplier: float) :
    width = int(image_src.shape[1] * multiplier)
    height = int(image_src.shape[0] * multiplier)
    dim = (width, height)
    image_result = cv.resize(image_src, dim)
    return image_result

def cdf(im) :
    c, b = cumulative_distribution(im)
    for i in range(b[0]) :
        c = np.insert(c, 0, 0)
    for i in range(b[-1] + 1, 256) :
        c = np.append(c, 1)
    return c

def hist_matching(c, c_t, im) :
    b = np.interp(c, c_t, np.arange(256))
    pix_repl = {i:b[i] for i in range(256)}
    mp = np.arange(0, 256)
    for (k,v) in pix_repl.items() :
        mp[k] = v
    s = im.shape
    im = np.reshape(mp[im.ravel()], im.shape)
    im = np.reshape(im, s)
    return im

if __name__ == "__main__" :
    
    color = ('r','g','b')
    
    img_h = 500
    img_w = 800
    
    img_1 = cv.cvtColor(cv.imread('canyon.png'), cv.COLOR_BGR2RGB);
    img_1 = crop(img_1, (img_h, img_w), origin = (0, 0))
    
    img_2 = cv.cvtColor(cv.imread('Nature.jpg'), cv.COLOR_BGR2RGB);
    img_2 = zoom(img_2, 2)
    img_2 = crop(img_2, (img_h, img_w), origin = (0, 0))
    
    # Image 1
    plt.subplot(321)
    plt.imshow(img_1)
    plt.title('')

    # Image 1 Histrogram
    plt.subplot(322)
    for i,col in enumerate(color):
        histr = cv.calcHist([img_1],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title('')
       
    # Image 2
    plt.subplot(323)
    plt.imshow(img_2)
    plt.title('')
    
    # Image 2 Histrogram
    plt.subplot(324)
    for i,col in enumerate(color):
        histr = cv.calcHist([img_2],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title('')


    # Image ใหม่
    img_ใหม่ = np.asarray(img_h * [img_w * [(0, 0, 0)]], np.uint8)
    
    for i in range(3) :
        c = cdf(img_1[..., i])
        c_t = cdf (img_2[..., i])
        img_ใหม่[..., i] = hist_matching(c, c_t, img_1[..., i])
    
    #Image ใหม่
    plt.subplot(325)
    plt.imshow(img_ใหม่)
    plt.title('Equalized Image')
    
    # Image ใหม่ Histrogram
    plt.subplot(326)
    for i,col in enumerate(color):
        histr = cv.calcHist([img_ใหม่],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title('')
    
    plt.tight_layout()
    plt.show()
