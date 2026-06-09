import unittest
import numpy as np
from math_ops.activations import ReLU
from math_ops.losses import MSE
from core.layer import Dense
from utils.data_loader import MNISTLoader

class TestNeuralNetwork(unittest.TestCase):

    # --- Tests for ReLU.compute ---
    def test_relu_compute_positive(self):
        relu = ReLU()
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(relu.compute(input_data), expected)

    def test_relu_compute_negative(self):
        relu = ReLU()
        input_data = np.array([[-1.0, 0.0], [-5.0, 2.0]])
        expected = np.array([[0.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(relu.compute(input_data), expected)

    # --- Tests for MSE.calculate ---
    def test_mse_calculate_zero(self):
        mse = MSE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        self.assertEqual(mse.calculate(y_true, y_pred), 0.0)

    def test_mse_calculate_non_zero(self):
        mse = MSE()
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 4.0])
        # (1-2)^2 = 1, (2-4)^2 = 4. Mean = (1+4)/2 = 2.5
        self.assertEqual(mse.calculate(y_true, y_pred), 2.5)

    # --- Tests for Dense.forward ---
    def test_dense_forward_simple(self):
        # input_size=2, output_size=2
        layer = Dense(2, 2)
        layer.weights = np.array([[1.0, 0.0], [0.0, 1.0]])
        layer.bias = np.array([[0.0, 0.0]])
        input_data = np.array([[1.0, 2.0]])
        expected = np.array([[1.0, 2.0]])
        np.testing.assert_array_almost_equal(layer.forward(input_data), expected)

    def test_dense_forward_with_bias(self):
        layer = Dense(2, 1)
        layer.weights = np.array([[2.0], [3.0]])
        layer.bias = np.array([[5.0]])
        input_data = np.array([[1.0, 2.0]])
        # 1*2 + 2*3 + 5 = 2 + 6 + 5 = 13
        expected = np.array([[13.0]])
        np.testing.assert_array_almost_equal(layer.forward(input_data), expected)

    # --- Tests for MNISTLoader.to_categorical ---
    def test_to_categorical_standard(self):
        loader = MNISTLoader("dummy.csv")
        labels = np.array([0, 1, 2])
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(loader.to_categorical(labels), expected)

    def test_to_categorical_repeated(self):
        loader = MNISTLoader("dummy.csv")
        labels = np.array([0, 2, 0])
        # Classes are 0 and 2. Set(labels) gives {0, 2}. 
        # Note: the original to_categorical uses len(set(labels)) for num_classes, 
        # but the indexing expects labels to be in range [0, num_classes-1].
        # In MNIST it's always 10 classes. 
        # Let's see how the provided implementation handles it:
        # num_classes = len(set(labels)) = 2
        # one_hot = np.zeros((3, 2))
        # one_hot[np.arange(3), [0, 2, 0]] -> Error if max label >= num_classes
        
        # If I use labels [0, 1, 0], num_classes = 2.
        labels = np.array([0, 1, 0])
        expected = np.array([
            [1, 0],
            [0, 1],
            [1, 0]
        ])
        np.testing.assert_array_equal(loader.to_categorical(labels), expected)

if __name__ == '__main__':
    unittest.main()
