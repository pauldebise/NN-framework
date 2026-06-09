import sys
from PyQt5.QtWidgets import QApplication
from core.network import Network, plot_history, visualize_predictions
from core.layer import Dense
from math_ops.losses import MSE, CategoricalCrossEntropy
from math_ops.activations import ReLU, Sigmoid, Softmax, ActivationLayer, LeakyReLU
from utils.data_loader import MNISTLoader
from app.app import App


model_name = r"best_model.json"
train_dataset_path = r"data/mnist_train.csv"
test_dataset_path = r"data/mnist_test.csv"
nb_epochs = 70
learning_rate = 0.05
batch_size = 256



network = Network()
network.add(Dense(input_size=784, output_size=64))
network.add(ActivationLayer(LeakyReLU()))
network.add(Dense(64, output_size=16))
network.add(ActivationLayer(LeakyReLU()))
network.add(Dense(16, output_size=10))
network.add(ActivationLayer(Softmax()))
network.compile(CategoricalCrossEntropy())
network.summary()



dataloader_train = MNISTLoader(train_dataset_path)
x_raw, y_raw = dataloader_train.load_data()
x_train = dataloader_train.normalize(x_raw)
y_train = dataloader_train.to_categorical(y_raw)

dataloader_test = MNISTLoader(test_dataset_path)
x_raw, y_raw = dataloader_test.load_data()
x_test = dataloader_test.normalize(x_raw)
y_test = dataloader_test.to_categorical(y_raw)
validation_data = (x_test, y_test)


history = network.fit(x_train, y_train, nb_epochs, learning_rate, batch_size, validation_data=validation_data)
network.save(model_name)


plot_history(history)
visualize_predictions(network, x_test, y_test, num_images=3)


app = QApplication(sys.argv)

window = App(model_path=model_name)
window.show()
sys.exit(app.exec_())