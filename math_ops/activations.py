from abc import ABC, abstractmethod
import numpy as np
from core.layer import Layer

class ActivationFunction(ABC):
    """
    Abstract class for activation functions.
    """
    @abstractmethod
    def compute(self, x):
        pass

    @abstractmethod
    def backward(self, x, output_gradient):
        pass


class ReLU(ActivationFunction):
    """Activation function Rectified Linear Unit."""
    def compute(self, x):
        return np.maximum(0, x)

    def backward(self, x, output_gradient):
        return output_gradient * np.where(x > 0, 1.0, 0.0)

class LeakyReLU(ActivationFunction):
    """Activation function Leaky Rectified Linear Unit."""
    def compute(self, x):
        return np.maximum(0.1*x, x)

    def backward(self, x, output_gradient):
        return output_gradient * np.where(x > 0, 1.0, 0.1)

class Sigmoid(ActivationFunction):
    """Activation function Sigmoid."""
    def compute(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, output_gradient):
        sigmoid = self.compute(x)
        return output_gradient * (sigmoid * (1 - sigmoid))
    
class Softmax(ActivationFunction):
    """Activation function Softmax."""
    def compute(self, x):
        x_norm = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x_norm) / np.sum(np.exp(x_norm), axis=1, keepdims=True)

    def backward(self, x, output_gradient):
        s = self.compute(x)
        return s * (output_gradient - np.sum(output_gradient * s, axis=1, keepdims=True))

    def derivative(self, x):
        softmax = self.compute(x)
        s = softmax.reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        return jacobian    



class ActivationLayer(Layer):
    """A layer without parameters that just applies a mathematical activation function."""
    def __init__(self, activation_strategy):
        super().__init__()
        self.__activation = activation_strategy
        self.input = None

    @property
    def activation(self):
        return self.__activation
    #No setter because activation should not be changed after layer creation.

    def forward(self, input_data):
        self.input = input_data
        return self.activation.compute(self.input)

    def backward(self, output_gradient, learning_rate):
        return self.activation.backward(self.input, output_gradient)
