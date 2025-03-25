
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder

from src.model.cnn import get_new_model
from src.model.utils import *

(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

# Normalizing the data so the color values lie in the interval [0, 1]:
train_data = train_data / 255.
test_data = test_data / 255.
# Verifying the shape of our data:
print(train_data.shape)
print(test_data.shape)

#These class names can be referenced in the dataset documentation
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
enc = OneHotEncoder()
transformed = enc.fit_transform(train_labels)
train_labels_encoded = transformed.toarray()
transformed = enc.fit_transform(test_labels)
test_labels_encoded = transformed.toarray()

# convert to 1 channel
train_data = np.average(train_data, axis=-1)
test_data = np.average(test_data, axis=-1)

train_data = train_data[...,np.newaxis] 
test_data = test_data[...,np.newaxis]

# Select testing data 
one_image_data = train_data[0,...]
one_image_data = one_image_data[np.newaxis,...]

train_data = train_data[1:,...]
train_labels = train_labels[1:,...]
train_labels_encoded = train_labels_encoded[1:,...]
     
model_benchmark = get_new_model(input_shape=(32,32,1))
model_benchmark.summary()

saving_direc = "./outputs/"
checkpoint_best_epoch = get_checkpoint_best_epoch(f'{saving_direc}Checkpoint_best_epoch_benchmark/checkpoint.weights.h5')
early_stopping = get_early_stopping()

callbacks = [checkpoint_best_epoch, early_stopping]

history_benchmark = model_benchmark.fit(x=train_data,
                                        y=train_labels_encoded,
                                        epochs=50,batch_size=100,
                                        validation_data=(test_data,test_labels_encoded),
                                        callbacks=callbacks)