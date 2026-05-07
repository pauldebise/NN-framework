from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    """
    Classe abstraite pour le pattern Strategy des fonctions de coût.
    """
    @abstractmethod
    def calculate(self, y_true, y_pred):
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass

class MSE(LossFunction):
    """Erreur Quadratique Moyenne (Mean Squared Error)."""
    def calculate(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def derivative(self, y_true, y_pred):
        n = np.size(y_true)
        return 2*(y_pred - y_true)/n

class CategoricalCrossEntropy(LossFunction):
    """Erreur d'Entropie Croisée"""
    def calculate(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return - (y_true / y_pred) / y_true.shape[0]
