import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.variables import DATA_PATH, MODEL_PATH


def load_or_initialize_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    """
    Load an existing model from disk or train a new model if none exists.
    """
    # Check if the model exists, if so, load it
    if os.path.exists(model_path):
        print('Loading existing model...')
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        print('Model loaded.')
    else:
        print('Model not found, training a new one...')
        # Train a new model using the dataset
        model = train_model(data_path)      
        # Save the trained model
        save_model(model, model_path)
    
    return model

def train_model(data_path=DATA_PATH):
    """
    Train a new Logistic Regression model on the dataset.
    """
    # Check if the dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found at {data_path}.')

    print('Loading dataset...')
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data['species']
    
    # Initialize and train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    print('Model trained.')
    
    return model

def save_model(model, model_path=MODEL_PATH):
    """
    Save the trained model to the specified path.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('Saving the trained model...')
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    print(f'Model trained and saved at {model_path}.')
