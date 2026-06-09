import unittest
import numpy as np
from math_ops.activations import ReLU
from math_ops.losses import MSE
from core.layer import Dense
from utils.data_loader import MNISTLoader

class TestNeuralNetwork(unittest.TestCase):

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

    def test_mse_calculate_zero(self):
        mse = MSE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        self.assertEqual(mse.calculate(y_true, y_pred), 0.0)

    def test_mse_calculate_non_zero(self):
        mse = MSE()
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 4.0])
        self.assertEqual(mse.calculate(y_true, y_pred), 2.5)

    def test_dense_forward_simple(self):
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
        expected = np.array([[13.0]])
        np.testing.assert_array_almost_equal(layer.forward(input_data), expected)

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
        labels = np.array([0, 1, 0])
        expected = np.array([
            [1, 0],
            [0, 1],
            [1, 0]
        ])
        np.testing.assert_array_equal(loader.to_categorical(labels), expected)

if __name__ == '__main__':
    unittest.main()
