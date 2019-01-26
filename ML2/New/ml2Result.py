#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# The neural network model into base_model
base_model = load_model('C:\DeepLearning\TrainingData\ML2\\resnet50_1547199544.h5')

# Cut the customized layers from the read resnet algorithm
x=base_model.get_layer('activation_48').output

# Eport network to one dimension
vector=Flatten()(x)

# Make the model with the whole network as input and the vector as output
model = Model(inputs=base_model.input, outputs=vector)


# In[4]:


test_datagen = ImageDataGenerator()     # Creating the Data generator
validation_data_dir = r'C:\DeepLearning\RealImages\Validation'

validation_generator = test_datagen.flow_from_directory(
                                        validation_data_dir,
                                        target_size=(128, 128),
                                        batch_size=1,
                                        color_mode='rgb',
                                        class_mode = "categorical",
                                        shuffle=True)
x_img, y_label = next(validation_generator);
features = model.predict(x_img)


# In[6]:


import numpy as np

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))
plt.subplot(1,1,1)
image = x_img
image= image.astype(int)
image = np.reshape(image, (128, 128, 3))
plt.imshow(image)


# In[7]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

knn_file = r'C:\DeepLearning\TrainingData\ML2\\knn.sav'

neigh = joblib.load(knn_file)


# In[8]:


labels = neigh.kneighbors(features, return_distance=False)


# In[10]:


import numpy as np

test_images_file = r'C:\DeepLearning\TrainingData\TestImages\\images.npy'

imgsList = np.load(test_images_file)


# In[13]:


plt.figure(figsize=(20, 20))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image = imgsList[labels[0][i]]
    image= image.astype(int)
    image = np.reshape(image, (128, 128, 3))
    plt.imshow(image)


# In[ ]:




