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
        2*(y_pred - y_true)/n