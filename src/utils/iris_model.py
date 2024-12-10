"""
iris_model.py

Author: Lander Combarro Exposito
Created: December 04, 2024
Last Modified: December 10, 2024

Description
-----------
This module handles the dataset preparation, model training, 
and model persistence for the Iris Dataset Classification API. 
It includes functions to download the dataset, preprocess it, 
train a new classification model, save it, and load it for predictions.
"""

import os
import pickle

import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def download_data(data_path):
    """
    Downloads the Iris dataset from scikit-learn and saves it as a CSV file.
    
    Notes
    -----
    - If the file already exists, no action is taken.
    - The function creates the necessary directories if they do not exist.
    """
    # Check if file already exists
    if os.path.exists(data_path):
        print(f"The data already exists in {data_path}. No action taken")
        return None

    print('Downloading data from scikit-learn...')
    iris = datasets.load_iris()
    df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df_iris['species'] = iris.target

    # Create directory if neccesary
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))
    
    # Save data locally
    df_iris.to_csv(data_path, index=False)
    print(f'Data downloaded and saved in {data_path}')
    
    return None


def load_or_initialize_model(data_path, model_path):
    """
    Loads an existing model from disk or trains a new one if no model is found.
    
    Notes
    -----
    - If the model file exists, it is loaded directly.
    - If the model file does not exist, the function trains a new model
      using the dataset and saves it to the specified path.
    """
    # Check if the model exists, if so, load it
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print('Model not found')
        X_train, X_val, y_train, y_val = load_data(data_path=data_path)
        model = train_model(X_train=X_train, y_train=y_train)      
        save_model(model, model_path)
    
    return model


def load_model(model_path):
    """
    Loads a pre-trained model from a pickle file. Returns the model.  
    
    Raises
    ------    
    - FileNotFoundError: If the specified model file does not exist.
    """
    if os.path.exists(model_path):
        print(f'Loading existing model from {model_path}...')
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        print('Model loaded')

    return model


def load_data(data_path):
    """
    Loads the Iris dataset from a CSV file and splits it into training and validation sets.

    Raises
    ------
    - FileNotFoundError: If the dataset file does not exist.
    """
    # Check if the dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found in {data_path}')

    print(f'Loading {data_path} dataset...')
    df = pd.read_csv(data_path)
    print(f'Dataset loaded')
    X = df.iloc[:, :-1]
    y = df['species']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val

    
def train_model(X_train, y_train):
    """
    Trains a new classification model using the provided training data.

    Notes
    -----
    - The function currently uses a RandomForestClassifier with
      a minimum leaf sample size of 30.
    - Model training can be adapted by modifying the classifier.
    """
    print('Training new model...')    
    # Initialize and train the model
    # model = LogisticRegression(max_iter=200)
    model = RandomForestClassifier(min_samples_leaf=30)     # Not hyperparameter-optimized
    model.fit(X_train, y_train)
    print('Model trained')
    
    return model


def save_model(model, model_path):
    """
    Saves the trained model to a pickle file.

    Notes
    -----
    - Creates necessary directories if they do not exist.
    - Overwrites the existing file at the specified path if it exists.
    """
    print('Saving the trained model...')
    
    # Create the directory if it doesn't exist
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f'Model trained and saved in {model_path}')
    
    return None
