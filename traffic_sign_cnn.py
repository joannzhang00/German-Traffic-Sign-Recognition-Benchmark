#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os
from keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[2]:


def grayscale(img):
    grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    return grayimg
    
def standardize(img):
    pixels = np.asarray(img)
    # convert from integers to floats
    pixels = pixels.astype('float64')
    # compute mean std
    mean, std = pixels.mean(), pixels.std()
    # global standardization of pixels
    pixels = (pixels - mean) / std
    
    return pixels

def normalize(image):
    image_data = np.array(image)
    # normalize the data by the formula
    image_new = np.divide((image_data - 128),128)

    return image_new


# In[3]:


# Reading the input images and resize them to get x_train data
trainData = []
x_train_labels = []

height = 32
width = 32
channels = 3
classes = 43
n_inputs = height * width * channels

for i in range(classes) :
    path = os.path.join(os.getcwd(), 'gts', 'Train', str(i))
    Class=os.listdir(path)
    for pic in Class:
        try:
            image = cv2.imread(os.path.join(path, pic))
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            #  get the gray mode of the image
            image = grayscale(np.array(size_image))
            pixels = np.asarray(image)
            # convert from integers to floats
            pixels = pixels.astype('float32')
            pixels = pixels.reshape(32, 32, 1)

            #  standardize the image data
            #  image_from_array = standardize(image)
            trainData.append(pixels)
            x_train_labels.append(i)
        except:
            print('')
            
trainData = np.array(trainData)
x_train_labels = np.array(x_train_labels)

#Randomize the order of the input images
trainNum = np.arange(trainData.shape[0])

np.random.seed(43)
np.random.shuffle(trainNum)
x_train = trainData[trainNum]
y_train = x_train_labels[trainNum]


# In[4]:


print("Original x shape: ", x_train.shape)
print("x data type:", x_train.dtype)
print("y shape: ", y_train.shape)


# In[5]:


# split x_train data into train and validation
(x_train, x_val) = x_train[(int)(0.2 * len(y_train)):], x_train[:(int)(0.2 * len(x_train))]
x_train = x_train/255.
x_val = x_val/255.
(y_train,y_val) = y_train[(int)(0.2 * len(y_train)):],y_train[:(int)(0.2 * len(y_train))]

#Using one hote encoding for the train and validation labels
y_train = to_categorical(y_train, classes)
y_val = to_categorical(y_val, classes)


# In[6]:


print("Original x shape: ", x_train.shape)
print("x data type:", x_train.dtype)
print("y shape: ", y_train.shape)
print()
print("x_val shape: ", x_val.shape)
print("x_val data type:", x_val.dtype)
print("y_val shape: ", y_val.shape)


# In[25]:


#read the test data
csv_path = os.path.join(os.getcwd(), 'gts', 'Test.csv')
y_test_df = pd.read_csv(csv_path)
image_labels = y_test_df['Path'].to_numpy()
y_test_value = y_test_df['ClassId'].values

#Using one hote encoding for the test labels
y_test = to_categorical(y_test_value, classes)

test_data = []

for name in image_labels:
    try:
        test_path = os.path.join(os.getcwd(), 'gts', 'Test')
        image = cv2.imread(os.path.join(test_path, name.replace('Test/', '')))
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        # get the gray mode of the image
        image = grayscale(np.array(size_image))
        pixels = np.asarray(image)
        # convert from integers to floats
        pixels = pixels.astype('float32')
        pixels = pixels.reshape(32, 32, 1)

        test_data.append(pixels)
    except:
        print('')

x_test = np.array(test_data)
x_test = x_test/255.


# In[26]:


print("x_test shape: ", x_test.shape)
print("x_test type:", x_test.dtype)
print("y_test shape: ", y_test.shape)


# In[52]:


import seaborn as sns
from matplotlib import rcParams

# font size
sns.set(font_scale = 3)
# plot
sns.countplot(x_train_labels, color = 'royalblue')
plt.xlabel('Class ID')
plt.ylabel('Class')
plt.title('Train Classes Distribution')
# figure size in inches
rcParams['figure.figsize'] = 50, 15

plt.savefig('train_class_dist')
plt.show()


# In[9]:


# Number of training examples
n_train = x_train.shape[0]

# Number of validation examples
n_validation = x_val.shape[0]

# Number of testing examples
n_test = x_test.shape[0]

# the shape of an traffic sign image
image_shape = x_train[0].shape

# unique classes/labels there are in the dataset
n_classes = len(set(list(x_train_labels)))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# In[10]:


from keras.regularizers import l2

# Construct a CNN network
epochs = 20
batch_size = 512
dropout = 0.3
lrate = 0.001
mmt = 0.8


model = Sequential([
        Conv2D(filters=32, kernel_size=5, strides=1, activation='relu', padding='same', input_shape = (height, width, 1), kernel_initializer='he_uniform'),
        MaxPooling2D(2),
        Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),
        Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),
        Flatten(),
        Dense(256, activation = 'relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01)),
        Dropout(dropout),
        Dense(128, activation = 'relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01)),
        Dropout(dropout),
        Dense(64, activation = 'relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01)),
        Dropout(dropout),
        Dense(43, activation = 'softmax')])

# Get the summary of each layer
print(model.summary())

# sgd = keras.optimizers.SGD(lr = lrate, decay = 1e-4, momentum = mmt)

# model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

adam = keras.optimizers.Adam(lr = lrate)

model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])


# The final model consisted of the following layers:
# 
# | Layer         		|     Description	        						| 
# |:---------------------:|:-------------------------------------------------:|
# | Input  	       		| 32x32x1 Grayscale image   						| 
# | Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 		|
# | Max pooling	 	  	| 2x2 stride,  outputs 16x16x32 					|
# | Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 		|
# | Max pooling	 	  	| 2x2 stride,  outputs 8x8x64 						|
# | Convolution 5x5	 	| outputs 8x8x64      							|
# | Max pooling			| 2x2 stride,  outputs 4x4x64  						|
# | Fully connected 		| outputs 256      									|
# | Fully Connected		| outputs 128										|
# | Fully Connected		| outputs 64										|
# | Fully Connected		| outputs 43										|

# In[11]:


# train the model

history = model.fit(x_train, y_train, epochs = epochs, validation_data = (x_val, y_val))
# model.save('cnn_model_best.h5')

# Performance plot
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,2)
plt.show()

print("Training accuracy:")
model.evaluate(x_train, y_train)
print("Testing accuracy:")
model.evaluate(x_test, y_test)


# In[12]:


print("Training accuracy:")
model.evaluate(x_train, y_train)


# In[13]:


print("Testing accuracy:")
model.evaluate(x_test, y_test)


# In[16]:


# Performance plot
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,2)
plt.savefig('acc_loss')
plt.show()


# In[ ]:




