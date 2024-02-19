#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:13:24 2024

@author: wxdycq
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data augmentation generator for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=40,  # Range of random rotations
    width_shift_range=0.2,  # Range for horizontal shifts (as a proportion of total width)
    height_shift_range=0.2,  # Range for vertical shifts (as a proportion of total height)
    shear_range=0.2,  # Shear intensity (counter-clockwise shear angle)
    zoom_range=0.2,  # Range for random zoom
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Method to fill newly created pixels
)

# Define generator for test data (normalization only)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'Car_Dataset/Train'
test_dir = 'Car_Dataset/Test'

# Use train_datagen generator to load images from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Directory of training images
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=10,  # Batch size
    class_mode='binary'  # Binary classification
)

# Use test_datagen generator to load images from directory
validation_generator = test_datagen.flow_from_directory(
    test_dir,  # Directory of test images
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)
