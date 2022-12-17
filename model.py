# Importing Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from IPython.display import display
import cv2

# Sample Input 
# (Instead of doing the step below, we need to determine a way to build the dataloader pipeline)

img_gt_raw = Image.open('SIDD_Small_sRGB_Only/Data/0062_003_S6_03200_02500_4400_L/GT_SRGB_010.PNG')
img_n_raw = Image.open('SIDD_Small_sRGB_Only/Data/0062_003_S6_03200_02500_4400_L/NOISY_SRGB_010.PNG')

# The steps below are for image processing and should be incorporated in the dataloader

img_gt = img_gt_raw
img_n = img_n_raw

img_n = img_n.resize((128,128))
img_gt = img_gt.resize((128,128))

# Data Visualization skipped

# Graysacling
img_n = ImageOps.grayscale(img_n)
img_gt = ImageOps.grayscale(img_gt)

# Grayscale visualization skipped


# Converting the image to tesnors
img_n = tf.keras.utils.img_to_array(img_n)
#Normalize
img_n =  img_n/255.0

# Building Generator model
def generator_model():
    model = models.Sequential()
    model.add(layers.Conv2D(
        input_shape = (128,128,1), filters = 3, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1) ))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1)))
    # model.add(layers.BatchNormalization())
    # Will introduce more batch normalization later
    model.add(layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 8, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 3, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 1, kernel_size = 3, padding = 'same', strides = 1, activation = 'sigmoid'))
    return model    

generator = generator_model()
print(generator.summary)

# Simple Testing (Just done for 1 image; in actuality, we need to train the model which will be done through training loop)
generated_img = generator.predict(img_n)

# Visualizing the output from the generator in 1st forward pass (No training was done till now)
# cv2.imshow('generated_img', np.squeeze(generated_img))
# cv2.waitKey(0)

# Building Discrimator Model
def discriminator_model():
    model = models.Sequential()
    model.add(layers.Conv2D(
        input_shape = (128,128,1), filters = 3, kernel_size = 3, padding = 'same', strides = 1, activation = 'relu' ))
    # Will introduct dropout later to prevent overfitting and manage simultaneous learning of generator and discriminator
    # model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1)))
    model.add(layers.Conv2D(filters = 8, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1)))
    model.add(layers.Conv2D(filters = 3, kernel_size = 3, padding = 'same', strides = 1, activation = layers.LeakyReLU(0.1)))
    model.add(layers.Conv2D(filters = 1, kernel_size = 3, padding = 'same', strides = 1, activation = 'sigmoid'))
    return model

discriminator = discriminator_model()
print(discriminator.summary())
