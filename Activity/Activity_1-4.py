import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Swing.png', cv.IMREAD_GRAYSCALE)

# Resized
scale_percent = 60
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv.resize(img, dim)

#Plot
X, Y = np.mgrid[0:img.shape[0], 0:img.shape[1]]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('Height')
ax.set_ylabel('Width')
ax.set_zlabel('Color Intensity')
ax.plot_surface(X, Y, img, cmap='pink')

plt.show()