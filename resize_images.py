#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:59:33 2024

@author: wxdycq
"""

from PIL import Image
import os
from sklearn.model_selection import train_test_split
import shutil

def resize_images(image_dir, output_dir, new_size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        with Image.open(img_path) as img:
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img.save(os.path.join(output_dir, img_name))

resize_images('Car_Dataset/Sports_Car', 'Car_Dataset/Processed/Sports_Car')
resize_images('Car_Dataset/Offroad_Car', 'Car_Dataset/Processed/Offroad_Car')




# 定义训练和测试的目录路径
dataset_base_dir = 'Car_Dataset'
train_dir = os.path.join(dataset_base_dir, 'Train')
test_dir = os.path.join(dataset_base_dir, 'Test')

# 定义跑车和越野车的分类目录
categories = ['Sports_Car', 'Offroad_Car']

# 创建训练和测试目录及其子目录
for category in categories:
    # 创建训练目录
    train_category_dir = os.path.join(train_dir, category)
    os.makedirs(train_category_dir, exist_ok=True)
    
    # 创建测试目录
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

