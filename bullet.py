#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:55:04 2019

@author: phd
"""
import numpy as np
import pandas as pd
import os
import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Activation, ZeroPadding2D, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from models.vggnet import VGGNet
import matplotlib.pyplot as plt
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
dataset = '/path/Bullet_dataset'
modd = 'bullet.model'
label = 'mlb.pickle'
plot = 'BulletPlot.png'
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
BS =32
IMAGE_DIMS = (25,25,1)
EPOCHS = 20
INIT_LR = 1e-3

model = VGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes= 2,finalAct="sigmoid")
gen = ImageDataGenerator(rotation_range=25,width_shift_range=0.1,
                         height_shift_range=0.1,shear_range=0.2,
                         zoom_range=0.2,horizontal_flip=True,
                         fill_mode="nearest")
print("Data Generation [Traing, Validation and Testing]")
train_batches = gen.flow_from_directory(dataset+"/training",target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]),
                                        color_mode="grayscale",shuffle=True,seed=1,
                                        batch_size=32)
#valid_batches = gen.flow_from_directory(dataset+"/Validation", target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]), color_mode="grayscale", shuffle=True,seed=1,batch_size=32)
test_batches = gen.flow_from_directory(dataset+"/testing", target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]),
                                       shuffle=False,color_mode="grayscale",
                                       batch_size=16)
# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy",optimizer=opt, metrics=["accuracy"])
print("[training network.....]")
H = model.fit_generator(train_batches,steps_per_epoch=len(train_batches) // BS,
	epochs=EPOCHS, verbose=1)
no_steps = len(test_batches)
# save the model to disk
print("[INFO] serializing network...")
model.save(modd)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(plot)
