# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
It involves achieving high accuracy in classifying malaria-infected cells versus uninfected cells to aid in the diagnosis of malaria from microscopic images. Your task would be to optimize the model, possibly by tuning hyperparameters, trying different architectures, or using techniques like transfer learning to improve classification accuracy.

![image](https://github.com/Karthikeyan21001828/malaria-cell-recognition/assets/93427303/85c0633d-9df2-4d72-a44f-93c22163bd1f)

## Neural Network Model

![image](https://github.com/Karthikeyan21001828/malaria-cell-recognition/assets/93427303/9ef16a59-612c-471d-b63f-5be6ce38a243)

## DESIGN STEPS

### STEP 1:
Import necessary libraries for data manipulation, visualization, and deep learning.
### STEP 2:
Set up TensorFlow session to dynamically allocate GPU memory and log device placement.
### STEP 3:
Define the directory paths for the dataset and inspect their contents.
### STEP 4:
Load sample images from both classes (infected and uninfected) for visualization.
### STEP 5:
Explore image dimensions and distributions in the dataset using seaborn.
### STEP 6:
Define the image shape and construct a sequential model using Keras.
### STEP 7:
Add convolutional and pooling layers to the model architecture.
### STEP 8:
Flatten the layer and add dense layers with activation functions.
### STEP 9:
Compile the model specifying loss function, optimizer, and evaluation metrics.
### STEP 10:
Configure image data augmentation using ImageDataGenerator.
### STEP 11:
Set batch size and generate training and testing data batches.
### STEP 12:
Train the model on the training data for a specified number of epochs.
### STEP 13:
Evaluate the model's performance on the test data and visualize training history.
### STEP 14:
Generate predictions on the test data and calculate classification metrics.
### STEP 15:
Use random image selection for inference and display the prediction result.






## PROGRAM

### Name: Sanjay.R

### Register Number: 212222220038


```python
# to share the GPU
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix

my_data_dir = './dataset/cell_images'
os.listdir(my_data_dir)

test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'

os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[200]

para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[200])

plt.imshow(para_img)

# Checking the image dimensions
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
help(ImageDataGenerator)

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)

model = models.Sequential()

# Add convolutional layers
model.add(layers.Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten the layer
model.add(layers.Flatten())

# Add a dense layer
model.add(layers.Dense(128, activation='relu'))

# Output layer
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

batch_size = 16
help(image_gen.flow_from_directory)
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
train_image_gen.class_indices

results = model.fit(train_image_gen,epochs=3,
                              validation_data=test_image_gen
                             )

losses = pd.DataFrame(model.history.history)
print("Sanjay.R\n212222220038\n")
losses[['loss','val_loss']].plot()

model.metrics_names

print("Sanjay.R\n212222220038\n")
model.evaluate(test_image_gen)

pred_probabilities = model.predict(test_image_gen)

print("Sanjay.R\n212222220038\n")
test_image_gen.classes

predictions = pred_probabilities > 0.5
print("Sanjay.R\n212222220038\n")
print(classification_report(test_image_gen.classes,predictions))

print("Sanjay.R\n212222220038\n")
confusion_matrix(test_image_gen.classes,predictions)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-05-07 113515](https://github.com/AbishekAnand15/malaria-cell-recognition/assets/118706942/8d5aebb7-a351-43bb-99a1-eed77eb2db0a)

### Classification Report

![Screenshot 2024-05-07 113523](https://github.com/AbishekAnand15/malaria-cell-recognition/assets/118706942/2d24d169-ada1-40b1-8a7a-b06579cebde5)

### Confusion Matrix

![Screenshot 2024-05-07 113528](https://github.com/AbishekAnand15/malaria-cell-recognition/assets/118706942/6121eeb1-9ee8-4704-9f62-c2ac8f9defa5)

### New Sample Data Prediction

![Screenshot 2024-05-07 113534](https://github.com/AbishekAnand15/malaria-cell-recognition/assets/118706942/be72e11e-6cd8-4b9b-ba0c-746b7fc54513)

## RESULT
Thus, a deep neural network for Malaria infected cell recognition is developed and the performance is analyzed.
