# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:22:18 2019
Not: 2. dense layer da dropout kaldırıldı.
@author: Yasir
"""

import os
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D

model_kayıt= 'drive/proje/ModelKayıt'
kayıt = os.path.join(model_kayıt, 'kayıt3.h5')
dosyalar = 'drive/proje/dataset-resized'

x_train_data = os.path.join(dosyalar, 'x_train.npy')
x_test_data = os.path.join(dosyalar, 'x_test.npy')
y_train_data = os.path.join(dosyalar, 'y_train.npy')
y_test_data = os.path.join(dosyalar, 'y_test.npy')

x_train = np.load(x_train_data)
x_test = np.load(x_test_data)
y_train = np.load(y_train_data)
y_test = np.load(y_test_data)


model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(384,512,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(384*512*3,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# Output Layer
model.add(Dense(6))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=150, verbose=1, validation_data=(x_test, y_test), shuffle=True)
model.save(kayıt)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])