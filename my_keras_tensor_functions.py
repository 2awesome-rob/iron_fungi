################################################################################
# Bob's functions for use in KAGGLE DATA PROJECTS
#
# Tools for working with Keras tensors and data. Often paired with
# my_kaggle_functions
#
#
################################################################################

import os, warnings
warnings.filterwarnings("ignore")

# Import required libraries and toolkits 
import numpy as np
import pandas as pd 

import tensorflow as tf

from scipy import stats

import math
import random
#import statsmodels.api as sm
#import statsmodels.formula.api as smf

import itertools
from tqdm import tqdm
from time import time

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_globals(seed: int = 80085, verbose: bool = True):
    """
    Set tf global variables and configurations for the project.
    Returns a Globals namedtuple: (device, cores)
    """

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)

    tf.config.run_functions_eagerly(True)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    plt.rc('figure', autolayout=True)
    plt.rc('axes', labelsize='small', titlesize=10, titlepad=4)
    plt.rc('image', cmap='copper')
    return AUTOTUNE

def _rescale(pic,n):
    rows, cols, _ = np.shape(pic)
    rows = int(n * int(rows/n)) # make sure rows are divisible by n
    cols = int(n * int(cols/n)) # make sure cols are divisible by n
    pic = pic[:rows,:cols]
    rows = int(rows/n)
    cols = int(cols/n)
    img = np.zeros((rows,cols,3),np.float64)
    for i in range(rows):
        for j in range(cols):
            a = int(i*n)
            b = int(i*n+n)
            if a == b: b+=1
            c = int(j*n)
            d = int(j*n+n)
            if c == d: d+=1
            img[i,j,0] = np.average(pic[a:b,c:d,0])  # Red Channel
            img[i,j,1] = np.average(pic[a:b,c:d,1])  # Green Channel
            img[i,j,2] = np.average(pic[a:b,c:d,2])  # Blue Channel
    return img

def _load_jpeg_as_tensor(path, size=128, channels=1):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=channels)
    if image.shape[0] != size or image.shape[1] != size:
        channels=1
    if channels == 1:
        image = tf.image.resize(image, size=[size, size])
        image = tf.expand_dims(image, axis=0)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)/255
    return image

def _load_jpeg_as_array(path, size=None, channels=None):
    image = plt.imread(path)
    if size is not None:
        image = _rescale(image, size)
    if channels < 2: 
        image = np.dot(image[...,:3], [1/3,1/3,1/3])
    return image

def load_train_sample_images(verbose=True, channels=1):
    img_files = []
    keys = []
    for dirname, _, filenames in os.walk('../input'):
        if "train" in dirname and filenames != []:
            keys.append(dirname.split("/")[-1])
            sampled = random.sample(filenames, 1)
            for filename in sampled:
                img_files.append(os.path.join(dirname, filename))

    images = {}
    for i, filename in enumerate(img_files):
        images[keys[i]] = _load_jpeg_as_tensor(filename, channels=channels)
    print(f"Images loaded with keys: {list(images.keys())}")

    if verbose:
        plt.figure(figsize=(2*len(keys), 4))
        for i, key in enumerate(keys):
            plt.subplot( 1, len(keys)+1, i + 1)
            if channels==1:
                plt.imshow(tf.squeeze(images[key]).numpy(), cmap='grey')
            else:
                plt.imshow(tf.squeeze(images[key]).numpy())
            plt.axis('off')
            plt.title(f"Class: {key}")
        plt.tight_layout()   
        plt.show()
    return images

def load_dataset_from_path(path, batch=1, buffer=None):
    def _convert_to_float(image, label):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, label
    
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='binary',
        image_size=[128, 128],
        interpolation='nearest',
        batch_size=batch,
        shuffle=True,
    )

    if buffer is None:
        ds = (ds
              .map(_convert_to_float)
        )
    else:
        ds = (ds
              .map(_convert_to_float)
              .cache()
              .prefetch(buffer_size=buffer)
        )
    return ds

def plot_train_sample_thumbnails(samples=4):
    plot_files = []
    for dirname, _, filenames in os.walk('../input'):
        if "train" in dirname and filenames != []:
            sampled = random.sample(filenames, min(samples, len(filenames)))
            for filename in sampled:
                plot_files.append(os.path.join(dirname, filename))

    images = []
    for filename in plot_files[:40]:
        image = _load_jpeg_as_tensor(filename)
        images.append(image)

    plt.figure(figsize=(8, len(images)//2))
    for i, image in enumerate(images):
        plt.subplot(i//samples + 1, samples, i%samples + 1)
        plt.imshow(tf.squeeze(image), cmap='gray')
        plt.axis('off')
    plt.tight_layout()   
    plt.title("Sample Training Images")
    plt.show()

def get_basic_kernels():
    def _format_as_tensor(kern):
        kern = tf.cast(kern, dtype=tf.float32)
        kern = tf.reshape(kern, [*kern.shape, 1, 1])
        return kern

    kernels_dict={
        "edge": np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]),
        "bottom_sobel": np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]),
        "emboss": np.array([
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ]),
        "sharpen": np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        }

    kernels = {}
    for k in kernels_dict.keys():
        kernels[k] = _format_as_tensor(kernels_dict[k])

    print(f"Loaded {list(kernels.keys())} kernels")
    return kernels

def plot_data_augmentation(image, data_augmentation_layers = None):
    if data_augmentation_layers == None:
        data_augmentation_layers = [
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomContrast(0.5)
        ]

    def _data_augmentation(images):
        for layer in data_augmentation_layers:
            images = layer(images)
        return images

    plt.figure(figsize=(6, 6))
    for i in range(9):
        augmented_images = _data_augmentation(image)
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(tf.squeeze(augmented_images).numpy())
        plt.axis("off")
    plt.show()

def plot_image_kernels(image, kernels:dict):
    images = [image]
    titles = ["unfiltered"]
    #convolution filters
    for k in kernels.keys():
        image_filter = tf.nn.conv2d(
            input=image,
            filters=kernels[k],
            strides=1,
            padding='VALID',
        )
        images.append(image_filter)
        titles.append(k)

    cols = min(6, len(images))

    plt.figure(figsize=(2*cols, 12))
    for i, image in enumerate(images):
        plt.subplot(i//cols+1, cols, i%cols+1)
        if i==0: plt.imshow(tf.squeeze(image), cmap='gray')
        else: plt.imshow(tf.squeeze(image))
        plt.axis('off')
        plt.title(titles[i])
    plt.tight_layout()   
    plt.show()

def plot_image_cv_steps(image, kernel):
    image_filter = tf.nn.conv2d(
        input=image,
        filters=kernel,
        strides=1,
        padding='VALID',
    )

    image_detect = tf.nn.relu(image_filter)

    image_condense = tf.nn.pool(
        input=image_detect,
        window_shape=(2,2),
        pooling_type='MAX',
        strides=(1,1),
        padding='SAME',
    )

    images = [image, image_filter, image_detect, image_condense]
    titles = ['input', 'filter', 'detect', 'condense']

    plt.figure(figsize=(8, 8))
    for i, img in enumerate(images):
        plt.subplot(1, 4, i+1)
        if i==0: plt.imshow(tf.squeeze(img), cmap='gray')
        else: plt.imshow(tf.squeeze(img))
        plt.axis('off')
        plt.title(titles[i])
    plt.tight_layout()   
    plt.show()

def plot_thumbnails(samples=4):
    plot_files = []
    for dirname, _, filenames in os.walk('../input'):
        if "train" in dirname and filenames != []:
            sampled = random.sample(filenames, min(samples, len(filenames)))
            for filename in sampled:
                plot_files.append(os.path.join(dirname, filename))

    images = []
    for filename in plot_files:
        image = _load_jpeg_as_tensor(filename, channels=1)
        images.append(image)

    plt.figure(figsize=(8, 8))
    for i, image in enumerate(images):
        plt.subplot(i//samples +1, samples, i%samples +1)
        plt.imshow(tf.squeeze(image), cmap='gray')
        plt.axis('off')
    plt.tight_layout()   
    plt.show()

"""
TODO: kernel convolution filtering of time series:

detrend = tf.constant([-1, 1], dtype=tf.float32)
average = tf.constant([0.2, 0.2, 0.2, 0.2, 0.2], dtype=tf.float32)
spencer = tf.constant([-3, -6, -5, 3, 21, 46, 67, 74, 67, 46, 32, 3, -5, -6, -3], dtype=tf.float32) / 320


# UNCOMMENT ONE
#kernel = detrend
#kernel = average
kernel = spencer

# Reformat for TensorFlow
ts_data = machinelearning.to_numpy()
ts_data = tf.expand_dims(ts_data, axis=0)
ts_data = tf.cast(ts_data, dtype=tf.float32)
kern = tf.reshape(kernel, shape=(*kernel.shape, 1, 1))

ts_filter = tf.nn.conv1d(
    input=ts_data,
    filters=kern,
    stride=1,
    padding='VALID',
)

# Format as Pandas Series
machinelearning_filtered = pd.Series(tf.squeeze(ts_filter).numpy())

machinelearning_filtered.plot();

"""