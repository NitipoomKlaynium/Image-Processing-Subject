import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from keras.models import Model, Input
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import glob

face_dataset = "face_dataset/lfw/**/*.jpg"
face_mini = "face_mini/**/*.jpg"
filenames = glob.glob(face_mini)

all_images = []

for fname in filenames[:500] :
    img = image.load_img(fname, target_size=(100, 100), interpolation="nearest")
    img = image.img_to_array(img)
    img /= 255
    all_images.append(img)
all_images = np.array(all_images)

train_x, test_x = train_test_split(all_images, random_state=32, test_size=0.3)
train_x, val_x = train_test_split(train_x, random_state=32, test_size=0.3)

noise_factor = 0.4
Nmean = 0
Nstd = 1

x_train_noisy = train_x + (noise_factor * np.random.normal(loc=Nmean, scale=Nstd, size=train_x.shape) )
x_val_noisy = val_x + ( noise_factor * np.random.normal(loc=Nmean, scale=Nstd, size=val_x.shape) )
x_test_noisy = test_x + ( noise_factor * np.random.normal(loc=Nmean, scale=Nstd, size=test_x.shape) )


# Encoded
Input_img = Input(shape=(100, 100, 3))

x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
x2 = MaxPool2D( (2, 2))(x2)

encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)

# Decoded
x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x3 = UpSampling2D((2, 2))(x3)
x4 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
x5 = Conv2D(256, (3, 3), activation='relu', padding='same')(x4)
decoded = Conv2D(3, (3, 3), padding='same')(x5)

autoencoder = Model(Input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse') 
autoencoder.summary()

epoch = 2 # [2, 3, 4]
batch_size = 8 # [8, 16, 32 ]
callback = EarlyStopping(monitor='val_loss', mode='min')

history = autoencoder.fit(x_train_noisy, train_x, epochs=epoch, batch_size=batch_size, shuffle=True, validation_data=(x_val_noisy, val_x), callbacks=callback)

# predictions = autoencoder.predict(x_val_noisy)
predictions = autoencoder.predict(x_train_noisy)


amount = 5;

plt.subplot(4, 3, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


for i in range(amount) :
    plt.subplot(4, amount, i + 1 + amount)
    plt.axis('off')
    plt.imshow(x_train_noisy[i])
    plt.subplot(4, amount, i + 1 + amount * 2)
    plt.axis('off')
    plt.imshow(train_x[i])
    plt.subplot(4, amount, i + 1 + amount * 3)
    plt.axis('off')
    plt.imshow(predictions[i])

plt.show()