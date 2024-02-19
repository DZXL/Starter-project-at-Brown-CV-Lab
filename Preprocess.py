#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:02:53 2024

@author: wxdycq
"""

from sklearn.model_selection import train_test_split
import shutil
import os



# Define directory paths for training and testing
dataset_base_dir = 'Car_Dataset'
train_dir = os.path.join(dataset_base_dir, 'Train')
test_dir = os.path.join(dataset_base_dir, 'Test')

# Define the category of sports cars and SUVs
categories = ['Sports_Car', 'Offroad_Car']

# Create a training and testing directory and its subdirectories
for category in categories:
    # Create a training directory
    train_category_dir = os.path.join(train_dir, category)
    os.makedirs(train_category_dir, exist_ok=True)
    
    # Create a testing directory
    test_category_dir = os.path.join(test_dir, category)
    os.makedirs(test_category_dir, exist_ok=True)



def split_data(source_dir, train_dir, test_dir, test_size=0.2):
    images = os.listdir(source_dir)
    train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
    for img in train_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(train_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(test_dir, img))

split_data('Car_Dataset/Processed/Sports_Car', 'Car_Dataset/Train/Sports_Car', 'Car_Dataset/Test/Sports_Car')
split_data('Car_Dataset/Processed/Offroad_Car', 'Car_Dataset/Train/Offroad_Car', 'Car_Dataset/Test/Offroad_Car')
