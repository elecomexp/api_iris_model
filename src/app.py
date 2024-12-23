"""
app.py

Author: Lander Combarro Exposito
Created: December 04, 2024
Last Modified: December 10, 2024

Iris Model Prediction API
-------------------------
This Flask API provides functionality for predicting the species of an Iris flower
based on its sepal and petal measurements. It also offers functionality for 
retraining the underlying model and evaluating its accuracy.

Endpoints
---------
- GET /api/v1/predict: Predicts the Iris flower species based on input features (sepal_length, sepal_width, petal_length, petal_width).
- GET /api/v1/retrain: Re-trains the model with a new dataset and update the saved model.
- GET /api/v1/accuracy: Calculates and returns the accuracy of the current trained model on the validation dataset. 
- POST /webhook: Updates the model by pulling the latest changes from the GitHub repository (used for deployment).

Error Handling
--------------
The API also includes basic error handling for missing or invalid parameters, and ensures that the model is always available for predictions.
"""

# Libraries
import json
import os
import subprocess

from flask import Flask, jsonify, request
from sklearn.metrics import accuracy_score

from utils.iris_model import (download_data, load_or_initialize_model, load_data,
                              load_model, save_model, train_model)
from utils.variables import CLASS_MAPPING, DATA_PATH, MODEL_PATH

# Change the current working directory to the directory of this script
os.chdir(os.path.dirname(__file__))

# Create a Flask application instance and enable debug mode for development
app = Flask(__name__)
app.config['DEBUG'] = True

# Download data; and load or initialize model
download_data(data_path=DATA_PATH)
model = load_or_initialize_model(data_path=DATA_PATH, model_path=MODEL_PATH)

# Landing page route
@app.route('/', methods=['GET'])
def home():
    # Define the response dictionary with message and available endpoints
    response = {
        'message': 'Welcome to the Iris flower model prediction API',
        'endpoints': {
            '/api/v1/predict': 'Provides predictions based on input features (GET)',
            '/api/v1/retrain': 'Retrains the model with a new dataset (GET)',
            '/api/v1/accuracy': 'Shows the current accuracy of the model (GET)'
        },
        'example': {
            '/api/v1/predict?sepal_length=5.0&sepal_width=3.6&petal_length=1.4&petal_width=0.2': 
                'Add an endpoint like this one to predict the species of the Iris flower using the provided feature values'
        }
    }

    # Return the response as a JSON with proper formatting
    return app.response_class(
        response=json.dumps(response, ensure_ascii=False, indent=4),
        status=200,
        mimetype='application/json'
    )

# Perform prediction
@app.route('/api/v1/predict', methods=['GET'])
def predict():
    try:
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))
    except (TypeError, ValueError):
        return jsonify({'error': 
            'Please provide valid numeric values for all parameters: sepal_length, sepal_width, petal_length, and petal_width'
            }), 400

    try:
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        class_name = CLASS_MAPPING[int(prediction[0])]
    except Exception as e:
        # Handle unexpected errors during prediction
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

    # Return successful response with prediction result
    return jsonify({
        'prediction': class_name,
        'input': {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
    })

# Retrain the model with existing dataset and evaluate the new accuracy
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists(DATA_PATH):
        try:
            X_train, X_val, y_train, y_val = load_data(data_path=DATA_PATH)
            model = train_model(X_train=X_train, y_train=y_train)
            save_model(model=model, model_path=MODEL_PATH)
            
            accuracy = accuracy_score(y_true=y_val, y_pred=model.predict(X_val))
            return jsonify({'message': 'Model retrained successfully', 'accuracy': str(accuracy)})
            
        except Exception as e:
            return jsonify({'error': f'An error occurred during retraining: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Dataset for retraining not found'}), 404

# Calculate and return the accuracy of the current saved model on a validation dataset
@app.route('/api/v1/accuracy', methods=['GET'])
def accuracy():
    try:
        X_train, X_val, y_train, y_val = load_data(data_path=DATA_PATH)
        model = load_model(model_path=MODEL_PATH)
        accuracy = accuracy_score(y_true=y_val, y_pred=model.predict(X_val))
        return jsonify({'accuracy': str(accuracy)})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred during retraining: {str(e)}'}), 500

# # Simple webhook
# @app.route('/webhook', methods=['POST'])
# def webhook():
#     repo_path = '/home/elecomexp/api_iris_model'
#     server_wsgi = '/var/www/elecomexp_pythonanywhere_com_wsgi.py'

#     if request.is_json:
#         subprocess.run(['git', '-C', repo_path, 'pull'], check=True)
#         subprocess.run(['touch', server_wsgi], check=True)
#         return jsonify({'message': 'Deployment updated successfully'}), 200
#     else:
#         return jsonify({'error': 'Invalid request'}), 400

# Webhook
@app.route('/webhook', methods=['POST'])
def webhook():
    # Path to the repository where the pull will be performed
    path_repo = '/home/elecomexp/api_iris_model'
    server_wsgi = '/var/www/elecomexp_pythonanywhere_com_wsgi.py'

    # Check if the POST request contains JSON data
    if request.is_json:
        payload = request.json
        # Check if the payload contains repository information
        if 'repository' in payload:
            # Extract repository name and clone URL
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']

            # Change to the repository directory
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': 'The repository directory does not exist!'}), 404

            # Perform a git pull in the repository, trick to automatically reload PythonAnywhere WebServer
            try:
                subprocess.run(['git', 'pull', clone_url], check=True)
                subprocess.run(['touch', server_wsgi], check=True)  
                return jsonify({'message': f'A git pull was performed on the repository {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error performing git pull on the repository {repo_name}'}), 500
        else:
            return jsonify({'message': 'No repository information found in the payload'}), 400
    else:
        return jsonify({'message': 'The request does not contain JSON data'}), 400


# Main
if __name__ == '__main__':
    app.run()

