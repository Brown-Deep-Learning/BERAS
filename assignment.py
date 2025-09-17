from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
from visualize import visualize_predictions
import numpy as np

from beras.model import SequentialModel

def get_model() -> SequentialModel:
    model = SequentialModel(
        [
           # Add in your layers here as elements of the list!
           # e.g. Dense(10, 10),
        ]
    )
    return model

def get_optimizer():
    # choose an optimizer, initialize it and return it!
    return ...

def get_loss_fn():
    # choose a loss function, initialize it and return it!
    return ...

def get_acc_fn():
    # choose an accuracy metric, initialize it and return it!
    return ...

if __name__ == '__main__':

    ### Use this area to test your implementation!

    # 1. Create a SequentialModel using get_model

    # 2. Compile the model with optimizer, loss function, and accuracy metric
    
    # 3. Load and preprocess the data
    # NOTE: Make sure to one-hot encode your labels!
    
    # 4. Train the model

    # 5. Evaluate the model

    # 6. Call visualize_predictions to see your model's outputs! (Check out the function definition in visualize.py)
    