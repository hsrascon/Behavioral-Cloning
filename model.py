import os
import json
import pandas as pd
import numpy as np
import cv2

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# Read image data from csv file 
folder = './data/data/'
csv_path = 'driving_log.csv'

training_data = pd.read_csv(folder + csv_path, skipinitialspace=True)

print('Number of images per camera:', len(training_data.index))
print('Total number of images:', 3 * len(training_data.index))
print('Number of left turns:', len(training_data[(training_data['steering'] < 0)]))
print('Number of right turns:', len(training_data[(training_data['steering'] > 0)]))
print('Number of frames with no turns', len(training_data[(training_data['steering'] == 0)]))


# Left, center, and right column headings in csv file
headings = ['left', 'center', 'right']

# Create array to hold images and steering angles
images = []
angles = []

# Go through each image in the left, center, and right column headings in the csv file 
for index in headings:
    for row in range(len(training_data.index)):
        
        # Read image
        image = mpimg.imread(folder + training_data[index].iloc[row])
        
        # Get steering angle of current image
        angle = training_data['steering'].iloc[row]
        
        # Add offset to steering angle for left and right camera images
        if index == 'left':
            angle += 0.20
        if index == 'right':
            angle -= 0.20
        
        #print(image.shape)
        #print(training_data[a].iloc[search])
        #print(angle)

        # Add random brightness to image
        image_bright = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_bright[:, :, 2] = image_bright[:, :, 2] * (0.25 + np.random.uniform())
        image_bright = cv2.cvtColor(image_bright, cv2.COLOR_HSV2RGB)

        #image_bright = cv2.cvtColor(image_bright, cv2.COLOR_RGB2HSV)
    
        # Crop image  
        image_crop = image_bright[55:135, :, :]
        #print(image.shape)
    
        # Resize image to 32x64x3
        image_resize = imresize(image_crop,(32,64,3))
        #print(image_new.shape)

        # Append results to the 2 arrays that hold images and steering angles respectively 
        images.append(image_resize)
        angles.append(angle)

#print(len(angles))

# Create array to hold images that are flipped horizontally 
flip_images = []

# Flip all images horizontally found in the images array and append them to flip_images array
for i in range(len(images)):
    flip_images.append(np.fliplr(images[i]))
#np.array(flip_images).shape
#np.array(angles).shape

# Flip the corresponding steering angle of the image also by multiplying by -1
flip_angles = np.array(angles)*-1
#print(flip_angles)

# Add results to the 2 arrays that hold images and steering angles respectively
images.extend(flip_images)
angles.extend(flip_angles)
#print(angles)

# Create training set X_train and y_train from the images and angles arrays 
X_train = np.array(images)
y_train = np.array(angles)

print('Shape of X_train: ', X_train.shape)
print('Shape of y_train:', y_train.shape)


# Normalize images using Min-Max scaling
def normalize(image_data):
    a = -0.5
    b = 0.5
    color_min = np.min(image_data)
    color_max = np.max(image_data)
    
    return a + (((image_data - color_min) * (b - a)) / (color_max - color_min))

X_train = normalize(X_train)

print('Images of training features are normalized.') 


# Shuffle the data to change the order
X_train, y_train = shuffle(X_train, y_train)

print('Training set is shuffled.')


# Split data into training and validation sets 
print('Number of training features to split: ', len(X_train))

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

print('Training and validation sets are split.') 
print('Number of training features: ', len(X_train))
print('Number of validation features ', len(X_validation))


# Establish model architecture
model = Sequential()

model.add(BatchNormalization(axis=1, input_shape=(32,64,3)))
model.add(Convolution2D(24, 3, 3, subsample=(2,2), border_mode='valid')) #input_shape=(32,64,3)))
model.add(Activation('relu'))

model.add(Convolution2D(36, 3, 3, subsample=(2,2), border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(48, 3, 3, subsample=(2,2), border_mode='valid'))
#model.add(Dropout(.5))
model.add(Activation('relu'))

model.add(Convolution2D(64, 2, 2, subsample=(1,1), border_mode='valid'))
#model.add(Dropout(.5))
model.add(Activation('relu'))

model.add(Convolution2D(64, 2, 2, subsample=(1,1), border_mode='valid'))
#model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dropout(.5))


model.add(Flatten())


model.add(Dense(100))
#model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dropout(.5))

model.add(Dense(50))
#model.add(Dropout(.5))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

model.summary()


# Create checkpoint to check performance of model at every epoch 
checkpoint_path="models/model-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=True, mode='auto')


# Compile model
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)

# Train model
model.fit(X_train, y_train, nb_epoch=10, batch_size=128, validation_data=(X_validation, y_validation), callbacks = [checkpoint])

# Save model to json file
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# Save weights to HDF5 file
model.save_weights("model.h5")
print("Saved model to disk")

