import numpy as np
import cv2 as cv
# from keras.preprocessing.image import ImageDataGenerator
import glob

# All_files = glob.glob("pexels/*/*.jpg")

# result = []

# for file in All_files :
#   try :
#     img = cv.imread(file)
#     if img.shape[0] % 96 == 0   and img.shape[1] % 96 == 0 :
#           result.append(file)
#   except :
#     print(file)

x = ['pexels\\Background\\pexels-anthony-132474.jpg', 'pexels\\Background\\pexels-bruno-thethe-1910225.jpg', 'pexels\\Background\\pexels-dom-j-310452.jpg', 'pexels\\Background\\pexels-faik-akmd-1025469.jpg', 'pexels\\Background\\pexels-ibadah-mimpi-3283186.jpg', 'pexels\\Background\\pexels-jeremy-bishop-2422915.jpg', 'pexels\\Background\\pexels-jess-bailey-designs-827106.jpg', 'pexels\\Background\\pexels-jess-bailey-designs-839442.jpg', 'pexels\\Background\\pexels-roberto-nickson-2559941.jpg', 'pexels\\Background\\pexels-sabel-blanco-1480523.jpg', 'pexels\\Background\\pexels-sam-willis-3934512.jpg', 
'pexels\\Background\\pexels-valeria-boltneva-101472.jpg', 'pexels\\Background\\pexels-victor-freitas-600114.jpg', 'pexels\\Healthy_Collection\\pexels-freestocksorg-128598.jpg', 'pexels\\Healthy_Collection\\pexels-gratisography-4621.jpg', 'pexels\\Healthy_Collection\\pexels-photomix-company-96620.jpg', 'pexels\\Healthy_Collection\\pexels-photomix-company-96974.jpg', 'pexels\\Healthy_Collection\\pexels-pixabay-236813.jpg', 'pexels\\Healthy_Collection\\pexels-trang-doan-863998.jpg', 'pexels\\SeaScape\\pexels-dominika-roseclay-977737.jpg', 'pexels\\SeaScape\\pexels-eli-burdette-762528.jpg', 'pexels\\SeaScape\\pexels-nuno-obey-127160.jpg', 'pexels\\SeaScape\\pexels-pixabay-414247.jpg', 'pexels\\SeaScape\\pexels-porapak-apichodilok-346887.jpg', 'pexels\\SeaScape\\pexels-riccardo-307008.jpg', 'pexels\\SeaScape\\pexels-sebastian-voortman-165505.jpg', 'pexels\\SeaScape\\pexels-sebastian-voortman-189349.jpg', 'pexels\\SeaScape\\pexels-vincent-gerbouin-1167021.jpg', 'pexels\\SeaScape\\pexels-vincent-gerbouin-1172524.jpg']
print(*result, sep='\n')

# img2 = cv2.imread('Swing.png')
# # reshape ภาพ
# img2 = img2.reshape((1,) + img2.shape)
# print(img2.shape)
# fill_method = ['constant', 'nearest','reflect', 'wrap']

# Npic = 5
# rotation_range = 10
# width_shift_range = 0.3
# height_shift_range = 0.3
# shear_range = 0.3
# zoom_range = 0.1
# horizontal_flip = True
# out = cv2.VideoWriter('activity7.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 2, (img2.shape[2], img2.shape[1]))

# for m in fill_method :
#   datagen = ImageDataGenerator( rotation_range = rotation_range,
#                               width_shift_range = width_shift_range,
#                               height_shift_range = height_shift_range,
#                               shear_range = shear_range,
#                               zoom_range = zoom_range,
#                               horizontal_flip = horizontal_flip,
#                               fill_mode = m)
#   pic = datagen.flow(img2, batch_size= 1)

#   for i in range(Npic):
#     batch = pic.next()
#     im_result = batch[0].astype('uint8')
#     out.write(im_result)
# out.release()

