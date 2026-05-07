from core.layer import Layer
from math_ops.losses import LossFunction, MSE, CategoricalCrossEntropy
from core.layer import Dense
from math_ops.activations import ReLU, Sigmoid, Softmax, ActivationLayer, LeakyReLU
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
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

    def _generate_batches(self, X, y, batch_size, drop_last=True):
        """Helper that makes batches."""
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, num_samples, batch_size):
            if drop_last and i + batch_size > num_samples:
                break
            yield X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]

    def fit(self, x_train, y_train, epochs: int, learning_rate: float, batch_size: int = 32, validation_data=None):
        """
        Trains the network and returns a dictionary with training info.
        """
        if self.loss_function is None:
            raise ValueError("Le réseau n'est pas compilé.")

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        has_validation = validation_data is not None

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            samples_seen = 0

            batches = list(self._generate_batches(x_train, y_train, batch_size))
            pbar = tqdm(batches, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=True)

            for x_batch, y_batch in pbar:
                current_batch_size = x_batch.shape[0]
                samples_seen += current_batch_size

                predictions = self.forward(x_batch)

                loss = self.loss_function.calculate(y_batch, predictions)
                epoch_loss += loss * current_batch_size

                y_pred_classes = np.argmax(predictions, axis=1)
                y_true_classes = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(y_pred_classes == y_true_classes)

                gradient = self.loss_function.derivative(y_batch, predictions)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)

                running_loss = epoch_loss / samples_seen
                running_acc = epoch_correct / samples_seen
                pbar.set_postfix({'loss': f"{running_loss:.4f}", 'acc': f"{running_acc * 100:.2f}%"})

            train_loss = epoch_loss / samples_seen
            train_acc = epoch_correct / samples_seen
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)

            if has_validation:
                x_val, y_val = validation_data
                val_loss, val_acc = self.evaluate(x_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)

                pbar.set_postfix({
                    'loss': f"{train_loss:.4f}", 'acc': f"{train_acc * 100:.2f}%",
                    'val_loss': f"{val_loss:.4f}", 'val_acc': f"{val_acc * 100:.2f}%"
                })

        return history


    def evaluate(self, x_test, y_test):
        """
        Évalue les performances du réseau sur un jeu de données de test.
        Retourne la perte (loss) et la précision (accuracy globale).
        """
        if self.loss_function is None:
            raise ValueError("Le réseau n'est pas compilé.")

        predictions = self.forward(x_test)

        loss = self.loss_function.calculate(y_test, predictions)

        y_pred_classes = np.argmax(predictions, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        accuracy = np.mean(y_pred_classes == y_true_classes)

        return loss, accuracy

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

        if print_info:
            print("\nModel summary:")
            for layer_detail in layer_details:
                layer_name, nb_params, current_shape = layer_detail
                print(f"{layer_name:<20} {nb_params:<10} {current_shape}")
            print(f"Total number of parameters: {total_param}")

        return layer_details

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
                elif act_type == 'LeakyReLU':
                    strategy = LeakyReLU()
                else:
                    raise ValueError(f"Unknown activation function encountered during loading : {act_type}")

                self.add(ActivationLayer(strategy))

        print(f"Network loaded and rebuilt from {filepath}")


def plot_history(history):
    """Trace les courbes d'apprentissage (Loss et Accuracy)."""
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'b-', label='Training')
    if history['val_accuracy']:
        plt.plot(epochs, history['val_accuracy'], 'r--', label='Validation')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'b-', label='Training')
    if history['val_loss']:
        plt.plot(epochs, history['val_loss'], 'r--', label='Validation')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, x_data, y_data, num_images=5):
    """
    Plots random image examples and their predictions.
    """
    num_samples = x_data.shape[0]
    indices = np.random.choice(num_samples, num_images, replace=False)

    x_sample = x_data[indices]
    y_sample = y_data[indices]

    predictions = model.predict(x_sample)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_sample, axis=1)

    plt.figure(figsize=(10, 2.5 * num_images))

    for i in range(num_images):
        plt.subplot(num_images, 2, 2 * i + 1)
        image = x_sample[i].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        is_correct = predicted_classes[i] == true_classes[i]
        color = 'green' if is_correct else 'red'
        confidence = predictions[i][predicted_classes[i]] * 100

        plt.title(f"Pred: {predicted_classes[i]} ({confidence:.1f}%)\nVrai: {true_classes[i]}", color=color)

        plt.subplot(num_images, 2, 2 * i + 2)
        probs = predictions[i] * 100

        bars = plt.bar(range(10), probs, color='gray')
        plt.xticks(range(10))
        plt.ylim([0, 105])
        plt.ylabel("Confiance (%)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        bars[true_classes[i]].set_color('green')
        if not is_correct:
            bars[predicted_classes[i]].set_color('red')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filepath = r"model_test.json"

    network = Network()
    network.add(Dense(input_size=784, output_size=64))
    network.add(ActivationLayer(LeakyReLU()))
    network.add(Dense(64, output_size=16))
    network.add(ActivationLayer(LeakyReLU()))
    network.add(Dense(16, output_size=10))
    network.add(ActivationLayer(Softmax()))

    network.save(filepath)
    n = Network()
    n.load(filepath)
    n.compile(CategoricalCrossEntropy())
    n.summary()

    dataloader_train = MNISTLoader(r"..\data\mnist_train.csv")
    x_raw, y_raw = dataloader_train.load_data()
    x_train = dataloader_train.normalize(x_raw)
    y_train = dataloader_train.to_categorical(y_raw)

    dataloader_test = MNISTLoader(r"..\data\mnist_test.csv")
    x_raw, y_raw = dataloader_test.load_data()
    x_test = dataloader_test.normalize(x_raw)
    y_test = dataloader_test.to_categorical(y_raw)
    validation_data = (x_test, y_test)

    history = n.fit(x_train, y_train, 300, 0.05, 256, validation_data=validation_data)
    plot_history(history)
    n.save(filepath)
    visualize_predictions(n, x_test, y_test, num_images=3)