import numpy as np
import pandas as pd
class MNISTLoader:
    """
    Helper class to load and pre-compute MNIST data.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self):
        """Loads MNIST csv dataset."""
        data = pd.read_csv(self.filepath)
        # The first column holds the label (the answer).
        labels = data.iloc[:, 0].values
        # Other columns represents pixels.
        images = data.iloc[:, 1:].values
        images = images.reshape(data.shape[0], 784)
        return images, labels

    def normalize(self, data):
        """Normalizes pixel values from [0; 255] to [0; 1]."""
        return (data/255.0).astype(np.float32)
        

    def to_categorical(self, labels):
        """Transforms scalar label to one-hot vectors for learning."""
        num_classes = len(set(labels))  # Number of classes (10 for MNIST)
        one_hot = np.zeros((labels.size, num_classes))
        one_hot[np.arange(labels.size), labels] = 1
        return one_hot