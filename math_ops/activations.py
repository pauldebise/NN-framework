from abc import ABC, abstractmethod
import numpy as np
from core.layer import Layer

class ActivationFunction(ABC):
    """
    Classe abstraite pour le pattern Strategy des fonctions d'activation.
    """
    @abstractmethod
    def compute(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class ReLU(ActivationFunction):
    """Fonction d'activation Rectified Linear Unit."""
    def compute(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1.0, 0.0)


class Sigmoid(ActivationFunction):
    """Fonction d'activation Sigmoïde."""
    def compute(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        #return np.divide(np.exp(-x), (1 + np.exp(-x))**2)
        sigmoid = self.compute(x)
        return sigmoid * (1 - sigmoid) #Equivalent mathématiquement et plus rapide

class Softmax(ActivationFunction):
    """Fonction d'activation Softmax."""
    def compute(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def derivative(self, x):
        pass


class ActivationLayer(Layer):
    """
    Couche qui applique une fonction d'activation aux données.
    Hérite de Layer et utilise la Composition avec ActivationFunction.
    """
    def __init__(self, activation_strategy: ActivationFunction):
        super().__init__()
        self.activation = activation_strategy

    def forward(self, input_data):
        pass

    def backward(self, output_gradient, learning_rate):
        pass