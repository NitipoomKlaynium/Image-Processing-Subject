import cv2 as cv
import numpy as np
from keras.models import Model, Input
from keras.preprocessing import image
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
import glob
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping

img_data_set = "D:/work/Stuff/image_data_set/*.jpg"
filenames = glob.glob(img_data_set)

all_images = []
H, W = 150, 150
ch = 3

for fname in filenames[0:2508] :
    img = image.load_img(fname, target_size=(H, W), interpolation="nearest")
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

noise_amount = 5;

plt.rcParams['figure.figsize'] = [22, 12]
for i in range(noise_amount) :
    plt.subplot(2, noise_amount, i + 1)
    plt.axis('off')
    plt.imshow(x_train_noisy[i])
    plt.title("Noise")
    
    plt.subplot(2, noise_amount, i + 1 + noise_amount)
    plt.axis('off')
    plt.imshow(train_x[i])
    plt.title("Original")
 
plt.show()

def create_model(optimizer='adam'):
    Input_img = Input(shape=(150, 150, 3))
    
    # encoder layers
    # encoder layers
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x2 = MaxPool2D( (2, 2))(x2)
    encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    
    # decoding architecture
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x3 = UpSampling2D((2, 2))(x3)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    decoded = Conv2D(3, (3, 3), padding='same')(x1)
    
    # construct the autoencoder model
    autoencoder = Model(Input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mean_squared_error'])
    
    return autoencoder

model = KerasRegressor(build_fn=create_model, epochs=2, batch_size=16, verbose=0)


optimizer = ['SGD', 'RMSprop', 'Adadelta', 'Adam']
batch_size = [8, 16, 32]
epochs = [2,4,6]

# Grid Search Parameter
param_grid = dict(  batch_size=batch_size,
                    epochs=epochs,
                    optimizer=optimizer)

grid_ = GridSearchCV(   estimator=model,
                        n_jobs=1,
                        verbose= 0,
                        cv=2,
                        param_grid = param_grid)

grid_result = grid_.fit(x_train_noisy, train_x)

print('Best params: ', grid_result.best_params_)
print('Best score: ', grid_result.best_score_)

# Create RandomizedSearchCV
random_search = {'optimizer': ['SGD', 'RMSprop', 'Adadelta', 'Adam'],
'batch_size': list(np.linspace(8, 64, 5, dtype = int)),
'epochs': list(np.linspace(1, 10, 4, dtype = int))}

grid_rand = RandomizedSearchCV( estimator=model,
                                n_jobs=1,
                                verbose= 0,
                                cv=2,
                                random_state = 10,
                                n_iter = 10,
                                param_distributions=random_search,)

grid_rand_result = grid_rand.fit(x_train_noisy, train_x)

print('Best params: ', grid_rand_result.best_params_)
print('Best score: ', grid_rand_result.best_score_)

rand_means = grid_rand_result.cv_results_['mean_test_score']
rand_stds = grid_rand_result.cv_results_['std_test_score']
rand_params = grid_rand_result.cv_results_['params']

y = np.arange(len(rand_params))

plt.rcParams['figure.figsize'] = [16, 12]
fig, ax = plt.subplots()
bar_means = ax.barh(y, rand_means, 0.8, label='')

ax.set_title('Error results from All combinations of grid parameters')
ax.set_yticks(y, rand_params)
ax.bar_label(bar_means)
ax.legend()

fig.tight_layout()

plt.show()