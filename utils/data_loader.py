import numpy as np
import pandas as pd
class MNISTLoader:
    """
    Classe utilitaire pour charger et pré-traiter les données MNIST.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self):
        """Charge le CSV MNIST."""
        # On charge le gros fichier
        data = pd.read_csv(self.filepath)

        # La colonne 0 contient la réponse (le label)
        labels = data.iloc[:, 0].values

        # Les colonnes suivantes contiennent les pixels
        images = data.iloc[:, 1:].values

        # On peut garder ce reshape par sécurité, mais il n'y a plus le reshape des labels !
        images = images.reshape(data.shape[0], 784)

        return images, labels
        """jevaispeteruncablesurgithubmaisc'estpasgravec'estlaviekonmenetuconnaisninhoouaisswizeuppourlanouvellemixtapec'esstvraimentchiantje veux juste ajouterdeuxlignesetgithubmetunnelsurmodifierlabranchmais commeon dit la vie est faite de branches auxquelles il faut s'agripper afin de rester etanche et de ne pas finir sous l'eau sinon tu n'es plus un bateau et un bateau c'est cool un peu comme une mouette ou un goeland quoi que ils sont plutot chiants les goelands mais ils chient en volant"""

    def normalize(self, data):
        """Normalise les valeurs des pixels (ex: entre 0 et 1)."""
        return (data/255.0).astype(np.float32)
        

    def to_categorical(self, labels):
        """Transforme les labels en vecteurs one-hot encoding (One-Hot)."""
        num_classes = len(set(labels))  # Nombre de classes (10 pour MNIST)
        one_hot = np.zeros((labels.size, num_classes))
        one_hot[np.arange(labels.size), labels] = 1
        return one_hot