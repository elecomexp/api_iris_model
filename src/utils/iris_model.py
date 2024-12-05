import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from variables import DATA_PATH, MODEL_PATH

def load_or_initialize_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    """
    Load an existing model from disk or train a new model if none exists.

    This function first checks if a model already exists at the given model_path.
    If the model exists, it will be loaded from disk. If it doesn't exist, the function
    will load the dataset from the specified data_path, train a new Logistic Regression model,
    and save the model to the specified model_path.

    Args
    ----
    - data_path (str)
        The file path to the dataset (CSV file).
    - model_path (str)
        The file path where the model will be saved or loaded from.

    Returns
    -------
    - model (LogisticRegression): The trained Logistic Regression model.
    """
    # Check if the model exists, if so, load it
    if os.path.exists(model_path):
        print('Loading existing model...')
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        print('Model loaded.')
    else:
        # If the model doesn't exist, check if the dataset exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Dataset not found at {data_path}.')
        print('Model not found, training a new one...')
        
        # Train a new model using the dataset
        model = train_model(data_path)
        
        # Save the trained model
        save_model(model, model_path)
    
    return model

def train_model(data_path=DATA_PATH):
    """
    Train a new Logistic Regression model on the dataset.

    This function loads the dataset from the provided data_path, splits it into features and target,
    initializes a Logistic Regression model, and trains it using the dataset.

    Args
    ----
    - data_path (str)
        The file path to the dataset (CSV file).

    Returns
    -------
    - model (LogisticRegression): The trained Logistic Regression model.
    """
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

    This function serializes the model using pickle and saves it to disk.

    Args
    ----
    - model (LogisticRegression): The trained model to be saved.
    - model_path (str): The file path where the model will be saved.
    """
    print('Saving the trained model...')
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    print('Model trained and saved.')
