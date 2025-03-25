import numpy as np

from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder

class Data():
    def __init__(self):

        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = cifar10.load_data()
        self._preprocess()
    
    def _preprocess(self):
        """
        Normalise input data and transform labels to one-hot vectors
        """
        train_data = self.train_data / 255.
        test_data = self.test_data / 255.

        enc = OneHotEncoder()
        transformed = enc.fit_transform(self.train_labels)
        self.train_labels_encoded = transformed.toarray()
        transformed = enc.fit_transform(self.test_labels)
        self.test_labels_encoded = transformed.toarray()

        # simple transformation from rgb to grey scale
        train_data = np.average(train_data, axis=-1)
        test_data = np.average(test_data, axis=-1)
        self.train_data = train_data[...,np.newaxis] 
        self.test_data = test_data[...,np.newaxis]
    
    def __getitem__(self, idx):

        return (self.train_data[idx], self.train_labels_encoded[idx]), (self.test_data[idx], self.test_labels_encoded[idx])
    
    def all(self):
        return (self.train_data, self.train_labels_encoded), (self.test_data, self.test_labels_encoded)


