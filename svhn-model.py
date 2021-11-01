# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:06:36 2021

@author: shikh
"""

import scipy.io 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

train = scipy.io.loadmat("train_32X32.mat")
test = scipy.io.loadmat("test_32X32.mat")

trainX, trainY = train['X'], train['y']
testX, testY = test['X'], test['y']

trainX = np.moveaxis(trainX,-1,0)
testX = np.moveaxis(testX,-1,0)

trainX = trainX/255
testX = testX/255

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.15, random_state=200)

datagen = ImageDataGenerator(rotation_range=8,
                             zoom_range=[0.95, 1.05],
                             height_shift_range=0.10,
                             shear_range=0.15)

model = keras.Sequential([keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)), 
                          keras.layers.BatchNormalization(), 
                          keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'), 
                          keras.layers.MaxPooling2D((2,2)), 
                          keras.layers.Dropout(0.5), 
                          keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'), 
                          keras.layers.BatchNormalization(),
                          keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                          keras.layers.MaxPooling2D((2, 2)),
                          keras.layers.Dropout(0.5),
                          keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
                          keras.layers.BatchNormalization(),
                          keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
                          keras.layers.MaxPooling2D((2, 2)),
                          keras.layers.Dropout(0.5),
                          keras.layers.Flatten(), 
                          keras.layers.Dense(512, activation='relu'), 
                          keras.layers.Dropout(0.4),
                          keras.layers.Dense(10, activation='softmax')]) 

optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=False)
model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                 metrics=['accuracy'])

model.summary()

callbacks = [keras.callbacks.ModelCheckpoint(filepath="conv.{epoch:d}", monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)]

history = model.fit(datagen.flow(trainX, trainY, batch_size=128),
                              epochs=90, validation_data=(valX, valY), callbacks=callbacks)

loss,accuracy = model.evaluate(testX, testY, batch_size=128)    

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["val_loss","loss"])
