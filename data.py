import scipy.io as scio
import numpy as np

#parameters
orig_size = 512
channels = 1


# Load the dataset
matfile = scio.loadmat('masktrain11_100f.mat')
imgs = matfile['rd_f100']
labels = matfile['cfar_f100']

# dataset h1cycleh2_det.mat
#matfile = scio.loadmat('h1cycleh2_det.mat')
#imgs = matfile['RD']
#labels = matfile['CFAR']


indx = int(0.2 * imgs.shape[0])

train_images = imgs[:-indx, 0:512, 0:512]
test_images = imgs[-indx:-1, 0:512, 0:512]

train_labels = labels[:-indx, 0:512, 0:512]
test_labels = labels[-indx:-1, 0:512, 0:512]


# Train set
train_images = train_images.reshape((train_images.shape[0], orig_size, orig_size, channels))
train_images = np.array(train_images, dtype=np.float32)

train_labels = train_labels.reshape((train_labels.shape[0], orig_size, orig_size, channels))
train_labels = np.array(train_labels, dtype=np.float32)


# Test set
test_images = test_images.reshape((test_images.shape[0], orig_size, orig_size, channels))
test_images = np.array(test_images, dtype=np.float32)

test_labels = test_labels.reshape((test_labels.shape[0], orig_size, orig_size, channels))
test_labels = np.array(test_labels, dtype=np.float32)
