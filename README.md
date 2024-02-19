# Simple Image Classifier for Vehicle Classification

## Project Overview

This project was developed as a part of the application process for the Brown Visual Computing Group.
The goal was to create a simple image classifier that can distinguish between sports cars and off-road vehicles.

## Dataset Description

The dataset consists of 44 images, with 22 images of sports cars and 22 images of off-road vehicles. 
The images were collected from various free image sources on the internet and are used purely for educational purposes.

## Data Augmentation

Due to the small size of the dataset, data augmentation techniques were applied to increase the diversity of the training set. 
The augmentation included random rotations, width and height shifts, shear transformations, zooming, and horizontal flips.

## Model Architecture

A Convolutional Neural Network (CNN) was employed for this classification task. 
The CNN architecture consists of several convolutional layers followed by max-pooling layers, a flattening layer, 
and dense layers with the final output layer utilizing a sigmoid activation function for binary classification.

## Training Process

The model was trained using 80% of the images, with the remaining 20% used as a test set. 
The training was performed with data augmentation to mitigate the risk of overfitting due to the limited number of training samples.

## Results and Conclusion

The classifier achieved an accuracy range between 80% to 100% on the test set. 
These results indicate that the model is sufficiently robust for the task of vehicle classification given the 
size and limitations of the dataset.
<img width="706" alt="image" src="https://github.com/DZXL/Starter-project-at-Brown-CV-Lab/assets/61910258/669d9bf7-d6b4-4294-9709-f1c91ab758ef">


## How to Run the Code

To run the code, follow these steps:

1. Ensure you have Python and the necessary libraries installed (TensorFlow, Keras, Pillow).
2. Clone the repository: `git clone <repository-url>`
3. Navigate to the cloned directory and run the scripts:
   - `python3 resize_images.py` to resize the images (if necessary)
   - `python3 CNN.py` to train and Evaluate the model

For more detailed instructions, please refer to the comments within each script.
