import numpy as np
import pandas as pd
class MNISTLoader:
    """
    Classe utilitaire pour charger et pré-traiter les données MNIST.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self):
        """Charge le CSV ou les binaires MNIST."""
        data = pd.read_csv(self.filepath)
        labels = data.iloc[:, 0].values
        images = data.iloc[:, 1:].values
        images = images.reshape(60000, 84)
        labels = labels.reshape(60000,10)
        """jevaispeteruncablesurgithubmaisc'estpasgravec'estlaviekonmenetuconnaisninhoouaisswizeuppourlanouvellemixtapec'esstvraimentchiantje veux juste ajouterdeuxlignesetgithubmetunnelsurmodifierlabranchmais commeon dit la vie est faite de branches auxquelles il faut s'agripper afin de rester etanche et de ne pas finir sous l'eau sinon tu n'es plus un bateau et un bateau c'est cool un peu comme une mouette ou un goeland quoi que ils sont plutot chiants les goelands mais ils chient en volant"""

        return images, labels

    def normalize(self, data):
        """Normalise les valeurs des pixels (ex: entre 0 et 1)."""
        return data/255.0
        

    def to_categorical(self, labels):
        """Transforme les labels en vecteurs one-hot encoding (One-Hot)."""
        num_classes = len(set(labels))  # Nombre de classes (10 pour MNIST)
        one_hot = np.zeros((labels.size, num_classes))
        one_hot[np.arange(labels.size), labels] = 1
