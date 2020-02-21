from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import argparse
from tensorflow import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import os
from network import *
from data import *

from numpy.random import seed
seed(1)

#Load model
model = Res_Unet()
#model.summary()

ap = argparse.ArgumentParser()
ap.add_argument("--weights","-w",required = False)
args = vars(ap.parse_args())


#load weights
if args["weights"] == "resunet_radar":
    model.load_weights('resunet_radar15.h5')

else:
    # fit the model
    results = model.fit(train_images, train_labels, validation_split=0.1, batch_size=8, epochs=20, callbacks=[tensorboard_callback])

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(test_images, test_labels)

decoded_imgs = model.predict(test_images)


n = 5
plt.figure(figsize=(5, 3))
for i in range(1,n):
    # display original
    ax = plt.subplot(3, n, i)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.imshow(test_images[i*3].reshape(512, 512))
    plt.gray()
    ax.set_title('Range Doppler Images')
    ax.set_xlabel('Range')
    ax.set_ylabel('Doppler')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(3, n, i + n)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.imshow(decoded_imgs[i*3].reshape(512, 512))
    plt.gray()
    ax.set_title('Predicted Images')
    ax.set_xlabel('Range')
    ax.set_ylabel('Doppler')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i + 2*n)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.imshow(test_labels[i*3].reshape(512, 512))
    plt.gray()
    ax.set_title('Ground Truth')
    ax.set_xlabel('Range')
    ax.set_ylabel('Doppler')
plt.show()
