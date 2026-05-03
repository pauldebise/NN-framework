import numpy as np
import pandas as pd
class MNISTLoader:
    """
    Classe utilitaire pour charger et pré-traiter les données MNIST.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self):
        """Charge le CSV ou les binaires MNIST."""
        data = pd.read_csv(self.filepath)
        labels = data.iloc[:, 0].values
        images = data.iloc[:, 1:].values
        return images, labels

    def normalize(self, data):
        """Normalise les valeurs des pixels (ex: entre 0 et 1)."""
        return data/255.0
        

    def to_categorical(self, labels):
        """Transforme les labels en vecteurs one-hot encoding (One-Hot)."""
        num_classes = len(set(labels))  # Nombre de classes (10 pour MNIST)
        one_hot = np.zeros((labels.size, num_classes))
        one_hot[np.arange(labels.size), labels] = 1
