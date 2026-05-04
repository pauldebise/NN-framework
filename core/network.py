from core.layer import Layer
from math_ops.losses import LossFunction, MSE, CategoricalCrossEntropy
from core.layer import Dense
from math_ops.activations import ReLU, Sigmoid, Softmax, ActivationLayer
import numpy as np
import json
from utils.data_loader import MNISTLoader


class Network:
    """
    Classe principale représentant le perceptron multicouche (MLP).
    Utilise la Composition : un réseau contient une liste de couches (Layer)
    et une fonction de perte (LossFunction).
    """
    def __init__(self):
        """Initialise un réseau vide."""
        self.layers = []
        self.loss_function = None

    def add(self, layer: Layer):
        self.layers.append(layer)

    def compile(self, loss_strategy: LossFunction):
        """Configure la fonction de perte du réseau."""
        self.loss_function = loss_strategy

    def forward(self, input_data):
        """Fait passer les données à travers toutes les couches."""
        output_data = input_data
        for layer in self.layers:
            output_data = layer.forward(output_data)
        return output_data

    def fit(self, x_train, y_train, epochs: int, learning_rate: float):
        """
        Entraîne le réseau de neurones sur un jeu de données.
        (Boucle sur les époques, calcul du forward, du loss, puis du backward).
        """
        for epoch in range(epochs):
            if self.loss_function is None:
                raise ValueError("Le réseau n'est pas compilé.")

            predictions = self.forward(x_train)
            error = self.loss_function.calculate(y_train, predictions)
            gradient = self.loss_function.derivative(y_train, predictions)
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, learning_rate)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {error:.4f} - Accuracy: {np.exp(-error)*100:.2f}%")

    def predict(self, x_test):
        """Prédit les sorties pour un jeu de données de test."""
        return self.forward(x_test)

    def summary(self, print_info: bool = True):
        total_param = 0
        current_shape = "Unknown"
        layer_details = []

        for layer in self.layers:
            layer_name = f"{layer.__class__.__name__}"

            if hasattr(layer, 'weights'):
                input_size, output_size = layer.weights.shape
                current_shape = f"(None, {output_size})"
                nb_params = layer.weights.size + layer.bias.size

            elif hasattr(layer, 'activation'):
                act_type = layer.activation.__class__.__name__
                layer_name = f"Activation ({act_type})"
                nb_params = 0

            else:
                nb_params = 0
                layer_name = f"Unknown"

            total_param += nb_params
            layer_details.append((layer_name, nb_params, current_shape))

        print("\nModel summary:")
        for layer_detail in layer_details:
            layer_name, nb_params, current_shape = layer_detail
            print(f"{layer_name:<20} {nb_params:<10} {current_shape}")
        print(f"Total number of parameters: {total_param}")

    def save(self, filepath: str):
        """
        Sauvegarde l'architecture et les poids du réseau dans un fichier JSON.
        (Figure imposée : Lecture/écriture de fichiers).
        """

        network_data = []
        for layer in self.layers:

            if hasattr(layer, "weights"):
                layer_data = {
                    'layer_type': 'Dense',
                    'input_size': layer.weights.shape[0],
                    'output_size': layer.weights.shape[1],
                    'weights': layer.weights.tolist(),
                    'bias': layer.bias.tolist()
                }
                network_data.append(layer_data)

            elif hasattr(layer, "activation"):
                layer_data = {
                    'layer_type': 'ActivationLayer',
                    'activation_type': layer.activation.__class__.__name__
                }
                network_data.append(layer_data)

        with open(filepath, 'w') as f:
            json.dump(network_data, f)

        print(f"Successfully saved {len(self.layers)} layers at {filepath}.")

    def load(self, filepath: str):
        """
        Charge l'architecture et les poids depuis un fichier JSON.
        """

        with open(filepath, 'r') as f:
            network_data = json.load(f)

        self.layers = []

        for data in network_data:

            if data['layer_type'] == 'Dense':
                layer = Dense(input_size=data['input_size'], output_size=data['output_size'])
                layer.weights = np.array(data['weights'])
                layer.bias = np.array(data['bias'])
                self.add(layer)

            elif data['layer_type'] == 'ActivationLayer':
                act_type = data['activation_type']

                if act_type == 'ReLU':
                    strategy = ReLU()
                elif act_type == 'Sigmoid':
                    strategy = Sigmoid()
                elif act_type == 'Softmax':
                    strategy = Softmax()
                else:
                    raise ValueError(f"Unknown activation function encountered during loading : {act_type}")

                self.add(ActivationLayer(strategy))

        print(f"Network loaded and rebuilt from {filepath}")


if __name__ == '__main__':
    filepath = r"modele_test.json"

    network = Network()
    network.add(Dense(input_size=784, output_size=64))
    network.add(ActivationLayer(Sigmoid()))
    network.add(Dense(64, output_size=10))
    network.add(ActivationLayer(Softmax()))

    network.save(filepath)
    n = Network()
    n.load(filepath)
    n.compile(CategoricalCrossEntropy())
    #n.compile(MSE())
    n.summary()
    dataloader = MNISTLoader(r"..\data\mnist_train.csv")
    x_raw, y_raw = dataloader.load_data()

    x_train = dataloader.normalize(x_raw)
    y_train = dataloader.to_categorical(y_raw)

    n.fit(x_train, y_train, 1000, 5)