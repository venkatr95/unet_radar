import tensorflow as tf
from losses import *
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

# Dimensions
raw_size = 512      #28 as original
adjust_size = 512    #28 as original
train_size = 512   #28 as original
channels = 1    #1 as original
base_number_of_filters = 32
kernel_size = (3, 3)
strides = (2, 2)
path = '/results'
time_string = time.strftime("%Y-%m-%d-%H-%M-%S")
results_path = os.path.join(path, time_string)


# Fixed model parameters
leak = 0.2
dropout_rate = 0.5


def next_power_2(n):
    count = 0
    # If it is a non-zero power of 2, return it
    if n and not (n & (n - 1)): 
        return n 
    # Keep dividing n by 2 until it is 0
    while n != 0: 
        n >>= 1
        count += 1
    # Result is 2 to the power of divisions taken
    return 1 << count
    
def mse(x1, x2, norm=2):
    return tf.reduce_mean(tf.square((x1 - x2) / norm))


def rmse(x1, x2, norm=2):
    return tf.sqrt(mse(x1, x2, norm))


def psnr(x1, x2, max_diff=1):
    return 20. * tf.log(max_diff / rmse(x1, x2)) / tf.log(10.)


def padding_power_2(shape):
    padded_size = next_power_2(max(shape))
    return ((padded_size - shape[0])//2, (padded_size - shape[1])//2)
    

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
def weighted_centropy_loss(inputs,outputs):
  return tf.nn.weighted_cross_entropy_with_logits(inputs,outputs,pos_weight=0.65)

def generate_and_save_images(model, epoch, test_inputs, test_labels):
    if model is None:
        predictions = test_inputs
    else:
        predictions = model(test_inputs, training=False)

    data = np.array([test_inputs, predictions, test_labels, predictions - test_labels])
    ntype = len(data)
    nrows = test_labels.shape[0]
    ncols = ntype
    fig = plt.figure(figsize=(8, 5))
    
    for i in range(ncols * nrows):
        plt.subplot(nrows, ncols, i+1)

        row = int(i / ncols)
        row_rel = row % ntype
        group = int(row / ntype)
        shift = ncols * (group * (ntype - 1) + row_rel)
        idx = i - shift

        for t in range(ntype):
            if row_rel == 0:
                j = int(i / ntype)
                rmse_cal = rmse(test_labels[j], predictions[j], norm=2)
                psnr_cal = psnr(test_labels[j], predictions[j], max_diff=1)
                #plt.xlabel('RMSE={:.3f}\nPSNR={:.2f}'.format(rmse_cal, psnr_cal), fontsize=4)
            if row_rel == t:
                plt.imshow(data[row_rel][idx, :, :, 0], interpolation='nearest', cmap='gray')
                break
        plt.xticks([])
        plt.yticks([])

    plt.savefig(os.path.join(results_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()    


def Res_Unet(pretrained_weights = None):
    """
    Network inspired from Micheal paper and used for target detection
    :return: residual UNet architecture of radar based target detection/segmentation
    """
    f = base_number_of_filters
    k = kernel_size
    s = strides
    sz = train_size
    c = channels
    pad = padding_power_2((sz, sz))

    inputs = tf.keras.layers.Input((sz, sz, c), name="ginput")
    inputs_pad = tf.keras.layers.ZeroPadding2D(pad, name="gpad")(inputs)
    # Encoder Part

    # encoder-block1
    ge1 = tf.keras.layers.Conv2D(4, k, padding="same", name="geconv11")(inputs_pad)
    ge1 = tf.keras.layers.LeakyReLU(leak, name="geact11")(ge1)
    ge1 = tf.keras.layers.Conv2D(4, k, padding="same", name="geconv12")(ge1)
    ge1 = tf.keras.layers.LeakyReLU(leak, name="geact12")(ge1)
    ge1 = tf.keras.layers.concatenate([ge1, inputs_pad], axis=3, name="gecat11")

    ge11 = tf.keras.layers.MaxPooling2D(2, name='pool1')(ge1)

    # encoder-block2
    ge2 = tf.keras.layers.Conv2D(8, k, padding="same", name="geconv21")(ge11)
    ge2 = tf.keras.layers.LeakyReLU(leak, name="geact21")(ge2)
    ge2 = tf.keras.layers.Conv2D(8, k, padding="same", name="geconv22")(ge2)
    ge2 = tf.keras.layers.LeakyReLU(leak, name="geact22")(ge2)
    ge2 = tf.keras.layers.BatchNormalization(name="gebn21")(ge2)
    ge2 = tf.keras.layers.concatenate([ge2, ge11], axis=3, name="gecat21")

    ge22 = tf.keras.layers.MaxPooling2D(2, name='poo22')(ge2)

    # encoder-block3
    ge3 = tf.keras.layers.Conv2D(16, k, padding="same", name="geconv31")(ge22)
    ge3 = tf.keras.layers.LeakyReLU(leak, name="geact31")(ge3)
    ge3 = tf.keras.layers.Conv2D(18, k, padding="same", name="geconv32")(ge3)
    ge3 = tf.keras.layers.LeakyReLU(leak, name="geact32")(ge3)
    ge3 = tf.keras.layers.BatchNormalization(name="gebn31")(ge3)
    ge3 = tf.keras.layers.concatenate([ge3, ge22], axis=3, name="gecat31")

    ge33 = tf.keras.layers.MaxPooling2D(2, name='poo33')(ge3)

    # encoder-decoder bottleneck
    ge4 = tf.keras.layers.Conv2D(32, k, padding="same", name="geconv41")(ge33)
    ge4 = tf.keras.layers.LeakyReLU(leak, name="geact41")(ge4)
    ge4 = tf.keras.layers.Dropout(dropout_rate, name="gddrop41")(ge4)
    ge4 = tf.keras.layers.Conv2D(32, k, padding="same", name="geconv42")(ge4)
    ge4 = tf.keras.layers.LeakyReLU(leak, name="geact42")(ge4)
    ge4 = tf.keras.layers.BatchNormalization(name="gebn41")(ge4)

    ge44 = tf.keras.layers.concatenate([ge4, ge33], axis=3, name="gecat41")

    gd1 = tf.keras.layers.Conv2DTranspose(16, k, s, padding="same", name="gdconv1")(ge44)
    gd1 = tf.keras.layers.LeakyReLU(leak, name="geact52")(gd1)

    # decoder-block3
    gd11 = tf.keras.layers.concatenate([gd1, ge3], axis=3, name="gdcat31")

    gd2 = tf.keras.layers.Conv2D(16, k, padding="same", name="gdconv31")(gd11)
    gd2 = tf.keras.layers.LeakyReLU(leak, name="gdact31")(gd2)
    gd2 = tf.keras.layers.Conv2D(16, k, padding="same", name="gdconv32")(gd2)
    gd2 = tf.keras.layers.LeakyReLU(leak, name="gdact32")(gd2)

    gd22 = tf.keras.layers.concatenate([gd2, gd11], axis=3, name="gdcat32")

    # decoder-block2
    gd3 = tf.keras.layers.Conv2DTranspose(8, k, s, padding="same", name="gdconv2")(gd22)
    gd3 = tf.keras.layers.LeakyReLU(leak, name="gdact21")(gd3)

    gd33 = tf.keras.layers.concatenate([gd3, ge2], axis=3, name="gdcat22")

    gd4 = tf.keras.layers.Conv2D(8, k, padding="same", name="gdconv21")(gd33)
    gd4 = tf.keras.layers.LeakyReLU(leak, name="gdact22")(gd4)
    gd4 = tf.keras.layers.Conv2D(8, k, padding="same", name="gdconv22")(gd4)
    gd4 = tf.keras.layers.LeakyReLU(leak, name="gdact23")(gd4)

    gd44 = tf.keras.layers.concatenate([gd4, gd33], axis=3, name="gdcat23")

    # decoder-block1
    gd5 = tf.keras.layers.Conv2DTranspose(4, k, s, padding="same", name="gdconv3")(gd44)
    gd5 = tf.keras.layers.LeakyReLU(leak, name="gdact11")(gd5)

    gd55 = tf.keras.layers.concatenate([gd5, ge1], axis=3, name="gdcat11")

    gd6 = tf.keras.layers.Conv2D(4, k, padding="same", name="gdconv11")(gd55)
    gd6 = tf.keras.layers.LeakyReLU(leak, name="gdact12")(gd6)
    gd6 = tf.keras.layers.Conv2D(4, k, padding="same", name="gdconv12")(gd6)
    gd6 = tf.keras.layers.LeakyReLU(leak, name="gdact13")(gd6)

    gd66 = tf.keras.layers.concatenate([gd6, gd55], axis=3, name="gdcat12")

    gd7 = tf.keras.layers.Conv2D(c, k, padding="same", activation="sigmoid",
                                          name="gdconvout")(gd66)

    outputs = tf.keras.layers.Cropping2D(pad, name="gcrop")(gd7)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="u_net")
    
    model.compile(optimizer ='adam', loss=[categorical_focal_loss(alpha=0.2, gamma=0)], metrics=['acc',f1_m,precision_m, recall_m]) #loss=[categorical_focal_loss(alpha=0.2, gamma=0)]

    return model
