import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

def initialize_model(train_path, model_path):
    """
    Load an existing model from disk or train a new model if none exists.
    """
    # Check if the model exists, if so, load it
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print('Model not found, training a new one...')
        # Train a new model using the dataset
        model = train_model(train_path)      
        # Save the trained model
        save_model(model, model_path)
    
    return model


def load_model(model_path):
    if os.path.exists(model_path):
        print('Loading existing model...')
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        print('Model loaded')
    return model


def load_data(data_path):
    # Check if the dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found at {data_path}')

    print(f'Loading {data_path} dataset...')
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1]
    y = df['species']
    return X, y
    
    
def train_model(train_path):
    """
    Train a new Logistic Regression model on the dataset.
    """
    X_train, y_train = load_data(train_path)
    
    # Initialize and train the model
    # model = LogisticRegression(max_iter=200)
    model = RandomForestClassifier(min_samples_leaf=30)
    model.fit(X_train, y_train)
    print('Model trained')
    
    return model


def save_model(model, model_path):
    """
    Save the trained model to the specified path.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('Saving the trained model...')
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f'Model trained and saved at {model_path}')
