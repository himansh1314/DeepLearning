# -*- coding: utf-8 -*-
#%%
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import matplotlib
import matplotlib.pyplot as plt
#%%
batch_size = 128

nb_classes = 10

nb_epoch = 12

img_rows, img_cols = 28, 28

nb_filters = 32
nb_pool = 2
nb_conv = 3
#%%
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0],1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape', X_train.shape)
print(X_train.shape[0], 'training sample')
print(X_test.shape[0], 'Test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
i = 4600
plt.imshow(X_train[i,0], interpolation='nearest')