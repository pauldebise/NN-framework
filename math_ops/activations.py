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
        
        sigmoid = self.compute(x)
        return sigmoid * (1 - sigmoid)
    
class Softmax(ActivationFunction):
    """Fonction d'activation Softmax."""
    def compute(self, x):
        x_norm = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x_norm) / np.sum(np.exp(x_norm), axis=1, keepdims=True)

    def derivative(self, x):
        softmax = self.compute(x)
        s = softmax.reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        return jacobian    



class ActivationLayer(Layer):
    def __init__(self, activation_strategy):
        super().__init__()
        self.activation = activation_strategy
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return self.activation.compute(self.input)

    def backward(self, output_gradient, learning_rate):
        if isinstance(self.activation, Softmax):
            s = self.activation.compute(self.input)
            return s * (output_gradient - np.sum(output_gradient * s, axis=1, keepdims=True))
        return output_gradient * self.activation.derivative(self.input)
