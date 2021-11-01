# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:19:36 2021

@author: shikh
"""


import tensorflow as tf

saved_model_dir = "conv.1"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
with open("svhn.tflite", "wb") as file:
    file.write(tflite_model)





