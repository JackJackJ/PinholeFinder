#Jack Lee
#jack.lee@rice.edu
#date of last revision: 7/12/2023
#This program will train a ML model to identify images based on three classes of optical artifact: pinhole, edge, and neither. 
#This model will be used to align a pinhole into the path of a laser beam
#inputs: training.tar.gz (training dataset)
#dependencies: matplotlib, numpy, tensorflow, PIL
#outputs: pinholeFinder.h5 (trained model)


## Setup
Import TensorFlow and other necessary libraries:
"""

import matplotlib.pyplot as plt #matplotlib will be used to visualize training results
import numpy as np #numpy will be used to visualize training results
import PIL #PIL, a drawing module, will be used to visualize training results
import tensorflow as tf #tensorflow will be used to create and train the image ID model

from tensorflow import keras #keras is the framework that will be doing the training
from tensorflow.keras import layers #the layers module in keras will be used to assist in training
from tensorflow.keras.models import Sequential #sequential is the type of machine learning model that we will be using

"""Import dataset (pinholes and nonpinholes)"""

import pathlib #pathlib will be used to import and initialize a path to the training dataset

dataset_url = "https://people.tamu.edu/~jackjackj/training.tgz" #this URL is a download link to a .tar.gz file that contains the complete training dataset
data_dir = tf.keras.utils.get_file('training.tar', origin=dataset_url, extract=True) #this will extract and locally download the URL
data_dir = pathlib.Path(data_dir).with_suffix('') #this creates a path to the extracted folder so that the program knows where to reference

"""Check number of images"""

image_count = len(list(data_dir.glob('*/*.jpg'))) #this will check if the dataset was downloaded successfully by getting the number of images
print(image_count) #print the number of images
print(data_dir) #print some info of the data directory

"""Create the dataset. Translating the folder to something the ML model will understand.

Start by making some parameters:
"""

batch_size = 32 #how much pictures does the model look at at a time
img_height = 180 #standardize the size of the images, if larger, linear (?) compress, if smaller, linear (?) expansion
img_width = 180 #standardize the size of the images

"""We will split up the dataset. 80% of pics will be used for training and 20% will be used for validation."""

train_ds = tf.keras.utils.image_dataset_from_directory( #gets the dataset and compeletes the translation
  data_dir, #parameter one: data directory we imported
  validation_split=0.2, #parameter two: I will be using 20% for validation 
  subset="training", #parameter three: this is a training set, so 80%
  seed=123, #parameter four: seed for random number generator
  image_size=(img_height, img_width), #parameter five: standardizing the images
  batch_size=batch_size) #parameter six: I set batch size 

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir, #parameter one: data directory we imported 
  validation_split=0.2, #parameter two: I will be using 20% for validation
  subset="validation", #parameter three: this is a validation set, so 20%
  seed=123, #seed for random number generator
  image_size=(img_height, img_width), #standardize
  batch_size=batch_size) #set batch size

class_names = train_ds.class_names #create an array containing the names of the possible classes (pinhole, edge, or neither)
print(class_names) #Print to check if i did that right

"""Visualizing the data:"""

for image_batch, labels_batch in train_ds: #another check to make sure I loaded stuff in the dataset correctly
  print(image_batch.shape) 
  print(labels_batch.shape)
  break

"""Configuring dataset for performance"""

AUTOTUNE = tf.data.AUTOTUNE #tensorflow buffer size used to limit overfitting

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #set buffer size for training
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) #set buffer size for validation

"""Standardize the RGB values:"""

normalization_layer = layers.Rescaling(1./255) #Normalize all colors to 255
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) #create new dataset with my normalized colors
image_batch, labels_batch = next(iter(normalized_ds)) #reapplying normalization to my image and labels
first_image = image_batch[0] #get the first image to check
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image)) #print my pixel values to check I did that right

"""Create the model:"""

num_classes = len(class_names) #get number of classes (3)

model = Sequential([ #initialize the neural network (Sequential model, trains again and again on data for as long as you tell it to)
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), #all lines are just image formatting
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

"""Compile the model"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) #form the neural network 
model.summary() #tell us about the model, confirm it compiled as expected

"""Train the model:"""

epochs=10 #feel free to play with this number to avoid over/undertraining
history = model.fit( #getting a record of training performance, training: model is told correct answer after submitting guess, validation: model gets correct answers only after all questions are answered. Create records of scores in both sets
  train_ds,
  validation_data=val_ds,
  epochs=epochs
) # Note model.fit is a function!

"""Predict two images: real pinhole screenshot and simulated edge screenshot. pinhole.jpg should be guessed as a pinhole and edge.jpg should be guessed as not a pinhole"""

pinhole_url = "https://people.tamu.edu/~jackjackj/pinhole.jpg" #real pinhole url
path = tf.keras.utils.get_file('pinhole', origin=pinhole_url) #make path
print(path) #print path
img = tf.keras.utils.load_img( #create image
    path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array) #pass image to my model
score = tf.nn.softmax(predictions[0]) #get the model's rating

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence." #tf is assigning confidence
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

edge_url = "https://people.tamu.edu/~jackjackj/edge.jpg"
path = tf.keras.utils.get_file('edge', origin=edge_url)

img = tf.keras.utils.load_img(
    path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

edge2_url = "https://people.tamu.edu/~jackjackj/edge1.jpg"
path = tf.keras.utils.get_file('edge2', origin=edge2_url)

img = tf.keras.utils.load_img(
    path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

edge_url = "https://people.tamu.edu/~jackjackj/edge2.jpg"
path = tf.keras.utils.get_file('edge2', origin=edge_url)

img = tf.keras.utils.load_img(
    path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

edge3_url = "https://people.tamu.edu/~jackjackj/edge3.jpg"
path = tf.keras.utils.get_file('edge3', origin=edge3_url)

img = tf.keras.utils.load_img(
    path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

#empty_url = "https://people.tamu.edu/~jackjackj/black.jpg"
#path = tf.keras.utils.get_file('black', origin=empty_url)

path = "622pinhole.jpg"
img = tf.keras.utils.load_img(
    path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
