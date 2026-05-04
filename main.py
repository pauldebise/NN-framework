import pandas as pd
import numpy as np
from core.layer import Dense,Layer
from utils.data_loader import MNISTLoader
from core.network import Network
from math_ops.activations import ReLU, Sigmoid, Softmax, ActivationLayer, ActivationFunction
from math_ops.losses import MSE
from tqdm import tqdm
import time

def evaluate_with_progress(model, X_test, y_test):
    predictions = []
    print("Évaluation du modèle...")
    
    
    for i in tqdm(range(len(X_test)), desc="Tests en cours", unit="img"):
        
        pred = model.predict(X_test[i]) 
        predictions.append(pred)
        
    
    accuracy = np.mean(np.array(predictions) == y_test)
    return accuracy


def main(): 
    loader_train = MNISTLoader("data/mnist_train.csv")
    loader_test = MNISTLoader("data/mnist_test.csv")
    X_train, Y_train = MNISTLoader.load_data(loader_train)
    X_test, Y_test = MNISTLoader.load_data(loader_test)
    
    model = Network()
    
    model.add(Dense(input_size=784, output_size = 128))
    model.add(ReLU())
    model.add(Dense(input_size=128, output_size=10))
    model.add(Softmax())
    model.compile(MSE())
    model.fit(X_train,Y_train,epochs = 40, learning_rate= 0.01)
    accuracy = evaluate_with_progress(model, X_test, y_test)
    print(f"Précision : {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()