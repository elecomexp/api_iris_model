
# Libraries
import json
import os
from flask import Flask, jsonify, request

from utils.iris_model import load_or_initialize_model, save_model, train_model
from utils.variables import CLASS_MAPPING, DATA_PATH, MODEL_PATH

# Change the current working directory to the directory of this script
os.chdir(os.path.dirname(__file__))

# Create a Flask application instance and enable debug mode for development
app = Flask(__name__)
app.config['DEBUG'] = True

# Function to load or initialize the model
model = load_or_initialize_model()

# Landing page route
@app.route('/', methods=['GET'])
def home():
    # Define the response dictionary with message and available endpoints
    response = {
        'message': 'Welcome to the Iris model prediction API',
        'endpoints': {
            '/api/v1/predict': 'Provides predictions based on input features (GET)',
            '/api/v1/retrain': 'Retrains the model with a new dataset (GET)'
        },
        'example': {
            '/api/v1/predict?sepal_length=5.0&sepal_width=3.6&petal_length=1.4&petal_width=0.2': 
                'Add an endpoint like this one to predict the species of the Iris flower using the provided feature values.'
        }
    }

    # Return the response as a JSON with proper formatting
    return app.response_class(
        response=json.dumps(response, ensure_ascii=False, indent=4),
        status=200,
        mimetype='application/json'
    )

@app.route('/api/v1/predict', methods=['GET'])
def predict():
    try:
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Please provide valid numeric values for all parameters: sepal_length, sepal_width, petal_length, and petal_width.'}), 400

    # Perform prediction
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

@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists(DATA_PATH):
        try:
            # Retrain the model with the existing dataset
            model = train_model(data_path=DATA_PATH)
            save_model(model, model_path=MODEL_PATH)
            
            # Optionally, evaluate the accuracy (uncomment if desired)
            # accuracy = accuracy_score(y_test, model.predict(X_test))
            # return jsonify({'message': 'Model retrained successfully', 'accuracy': accuracy})
            
            return jsonify({'message': 'Model retrained successfully'})
        except Exception as e:
            # Handle any errors during training or saving
            return jsonify({'error': f'An error occurred during retraining: {str(e)}'}), 500
    else:
        # Return error if dataset is not found
        return jsonify({'error': 'Dataset for retraining not found'}), 404


# # LUIS TAMAYO
# @app.route('/webhook', methods=['POST'])
# def webhook():
#     repo_path = '/home/LuTaOr/Despliegue_API'
#     server_wsgi = '/var/www/lutaor_pythonanywhere_com_wsgi.py'

#     if request.is_json:
#         subprocess.run(['git', '-C', repo_path, 'pull'], check=True)
#         subprocess.run(['touch', server_wsgi], check=True)
#         return jsonify({'message': 'Despliegue actualizado con éxito'}), 200
#     else:
#         return jsonify({'error': 'Solicitud no válida'}), 400


if __name__ == '__main__':
    app.run()

