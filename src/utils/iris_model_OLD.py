
import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

from variables import DATA_PATH, MODEL_PATH

# Function to load or initialize the model
def load_or_initialize_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    """
    Load an existing model from disk or train a new model if none exists.

    Args
    ----
    - data_path (str): Path to the dataset CSV file.
    - model_path (str): Path to the saved model file.

    Returns
    -------
    - model: A trained Logistic Regression model.
    """
    if os.path.exists(model_path):
        # Load the pre-trained model from disk
        print('Loading existing model...')
        model = pickle.load(open(model_path, "rb"))
        print('Model loaded.')
    else:
        # If the model doesn't exist but data does, train a new model and save it
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Dataset not found at {data_path}.')
        print('Model not found, training a new one...')
        
        # Load the dataset, prepare features and target, and initialize and train model 
        model = train_model(data_path)
        
        # Save the trained model
        save_model(model, model_path)
    
    return model

def train_model(data_path=DATA_PATH):
    # Load the dataset, prepare features and target, and initialize and train model 
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data['species']
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model

def save_model(model, model_path=MODEL_PATH):
    """
    Save the trained model
    """
    pickle.dump(model, open(model_path, "wb"))
    print('Model trained and saved.')
    
    
