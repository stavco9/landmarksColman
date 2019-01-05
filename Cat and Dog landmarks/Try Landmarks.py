#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# In[15]:


# You must add a r to the start of the string or else an error will occur
#DATADIR = r"C:\DeepLearning\Cat&Dog\PetImages"
DATADIR = r"C:\DeepLearning\FinalProject\Files"
CATEGORIES = ["12352", "12290"]

IMG_SIZE = 250  #Setting the constant size of each image
training_data = []

# This function read all the images to an array with constant size of 50x50 and adds to the array
# the class of the image (0 to dog and 1 to cat)
def create_training_data():
    count = 0
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to cats or dogs dir
        class_num = CATEGORIES.index(category)  #setting that 0 is a dog and 1 is a cat
        for img in os.listdir(path):
            try:
                #img_array = cv2.imread()
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                print("loading " + str(count) + " image of class " + str(class_num))
                count += 1
            except Exception as e:
                pass   # do nothing
            
create_training_data()    # Create the training data


# In[2]:


import random

# Shuffle all the loaded images
random.shuffle(training_data)


# In[3]:


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# We need to convert the List of images to a numpy array for Tensorflow
# We create a numpy array where the image is in the shape of (50,50,1). 1 since this is a grey scale
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print(X[0].shape)


# In[37]:



import pickle 

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[39]:


pickle_in = open("X.pickle", "rb")
#X = pickle.load(pickle_in)


# In[4]:


X[6]


# In[ ]:





# In[5]:


X = X / 255.0


# In[6]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import pickle


# In[7]:


# Build neural network
model = Sequential()

# Add convolutional layer of 64 convolution filter sized with 3x3 and the input shape is 50x50x1
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))

# The activation function is relu
model.add(Activation("relu"))

# The max pooling window is 2x2
model.add(MaxPooling2D(pool_size=(2,2)))


# In[51]:


X.shape[1:]


# In[52]:


# Add convolutional layer of 64 convolution filter sized with 3x3
model.add(Conv2D(64, (3,3)))

# The activation function is relu
model.add(Activation("relu"))

# The max pooling window is 2x2
model.add(MaxPooling2D(pool_size=(2,2)))

##################################################

#Flatening the data to one D
model.add(Flatten())

# 64 outputs
model.add(Dense(64))

# Adding relu activation method
model.add(Activation("relu"))

###################################################

#output layer 2 outputs
model.add(Dense(2))

# Softmax activation method
model.add(Activation('softmax'))


# In[53]:


model.summary()


# In[8]:


X.shape[1:]


# In[9]:


X.shape


# In[10]:


# Learning rate of 0.001 that presents a low back propogation
adamOptimizer = tf.keras.optimizers.Adam(lr=0.001)

model.compile(loss = 'sparse_categorical_crossentropy', 
             optimizer='adam',
             metrics=['accuracy'])


# In[11]:


import time

## 
## This class builds the model into class and saving it into a dictionary
##
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='C:\DeepLearning\FinalProject\logs1', **kwargs):
        NAME = "Cats-vs-dogss-ccn-64x2-{}".format(int(time.time()))
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, "%s_%s" % (NAME,"Training"))
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, "%s_%s" % (NAME,'validation'))

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', 'epoch_'): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


# In[56]:


# Fitting the data into 3 times (train the input into the maching learing neural network) and saving the training data to external folder
model.fit(X, y, batch_size=32, epochs = 3, validation_split=0.1, callbacks=[TrainValTensorBoard(write_graph=True)])


# In[12]:


# Measuring the loss
#train_loss, train_acc = model.evaluate(X,y)


# In[13]:


#train_loss


# In[14]:


#!tensorboard --logdir="C:\DeepLearning\FinalProject\logs1"


# In[16]:


import random

IMG_SIZE = 250  #Setting the constant size of each image
training_data = []
for category in CATEGORIES:
    count = 0;                
    path = os.path.join(DATADIR, category)  # path to cats or dogs dir
    class_num = CATEGORIES.index(category)  #setting that 0 is a dog and 1 is a cat
    for img in os.listdir(path):
        count = count + 1;               # increase the image count
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            training_data.append([img_array, class_num])
        except Exception as e:
            pass   # do nothing
        if count == 5:
            break;
        
random.shuffle(training_data)
testPairs = training_data;     # Getting a sample of 10 images

process_test =[cv2.resize(pair[0], (IMG_SIZE, IMG_SIZE))/255.0 for pair in testPairs] # resizing the images
test_labels = [pair[1] for pair in testPairs]
process_test= np.array(process_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)   # resize the array for the model
predictions = model.predict(process_test)


# In[17]:


import matplotlib.pyplot as plt

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap='gray')

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(CATEGORIES[predicted_label],
                                100*np.max(predictions_array),
                                CATEGORIES[true_label]),
                                color=color)

num_rows = 2
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, num_cols, i+1)
  plot_image(i, predictions, test_labels, [pair[0] for pair in testPairs])


# In[ ]:




