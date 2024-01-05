# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 17:40:29 2023

@author: karan
"""

"""# Test the working of GPU using a sample MNIST example.
"""


import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


# Listing Devices including GPU's with Tensorflow


from tensorflow.python.client import device_lib

device_lib.list_local_devices()


# To Check GPU in Tensorflow

tf.test.is_gpu_available()



# Load MNiST Dataset

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()



# Pre-processing of Training and Test Datasets¶

x_train, x_test = x_train / 255.0, x_test / 255.0


# Create Sequential Model Using Tensorflow Keras

# Architecture of the Network is :-

# 1). Input layer for 28x28 images in MNiST dataset

# 2). Dense layer with 128 neurons and ReLU activation function

# 3). Output layer with 10 neurons for classification of input images as one of ten digits(0 to 9)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


predictions = model(x_train[:1]).numpy()
predictions


# Creating Loss Function

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# Training and Validation

# The Model.fit method adjusts the model parameters to minimize the loss:

model.fit(x_train, y_train, epochs=5)


model.evaluate(x_test,  y_test, verbose=2)


