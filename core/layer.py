from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    """
    Classe abstraite représentant une couche générique du réseau de neurones.
    Toutes les couches doivent implémenter une propagation avant et arrière.
    """
    def __init__(self):
        self.input_data = None
        self.output_data = None

    @abstractmethod
    def forward(self, input_data):
        """Calcule la sortie de la couche en fonction de l'entrée."""
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        """
        Calcule le gradient par rapport à l'entrée et met à jour
        les paramètres de la couche si nécessaire.
        """
        pass


class Dense(Layer):
    """
    Couche complètement connectée (Fully Connected Layer).
    Hérite de Layer.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initialise les poids et les biais de manière aléatoire.
        """
        super().__init__()
        self.weights = (np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)).astype(np.float32)
        self.bias = np.zeros((1, output_size), dtype=np.float32)

    def forward(self, input_data):
        self.input_data = input_data
        return np.dot(input_data, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(self.input_data.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient