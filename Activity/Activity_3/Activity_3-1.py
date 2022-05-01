import cv2 as cv
import numpy as np
from matplotlib import image, pyplot as plt

def crop(image_src: np.ndarray, size: set, origin = (0, 0)) :
    img_result = image_src[origin[0]:(origin[0] + size[0]), origin[1]:(origin[1] + size[1])]
    return img_result

def zoom(image_src, multiplier: float) :
    width = int(image_src.shape[1] * multiplier)
    height = int(image_src.shape[0] * multiplier)
    dim = (width, height)
    image_result = cv.resize(image_src, dim)
    return image_result

if __name__ == "__main__" :
    
    # Resize image
    img = cv.imread('Swing.png')
    img = crop(img, (600, 600), (0, 50))

    # Generate weight array
    γ_array = [0.1]
    step = 0.025
    while(γ_array[-1] < 3.0) :
        γ_array += [γ_array[-1] + step]
        if (3.0 - γ_array[-1] < step) :
            γ_array += [3.0]
    a, b = 1.0, 0
    
    print(γ_array)
    
    # Create vdโอ writer
    vdโอ_fourcc = cv.VideoWriter_fourcc(*'mp4v')
    vdโอ = cv.VideoWriter("Swing_gamma.mp4", vdโอ_fourcc, 30.0, (600,600))
    
    # Frame writing
    
    for γ in γ_array :
        frame = np.uint8((a * ((img/255)**γ) + b) * 255)
        vdโอ.write(frame)
    
    for γ in γ_array[::-1] :
        frame = np.uint8((a * ((img/255)**γ) + b) * 255)
        vdโอ.write(frame)
        
    # vdโอ Release
    vdโอ.release()
 