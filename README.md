# API Iris Flower Model 

A simple lightweight API built with [Flask](https://flask.palletsprojects.com/en/stable/) for deploying a machine learning model trained on the Iris flower dataset. This project demonstrates how to integrate and serve ML models through a RESTful API, deployed on [PythonAnywhere](https://www.pythonanywhere.com/) for easy access.

## Features

- Predicts species of Iris flowers: setosa, versicolor, and virginica.
- RESTful endpoints for submitting sepal and petal measurements.
- Lightweight implementation suitable for quick prototyping.

## Notes on Model Performance

**The classification model used in this API is not optimized for maximum accuracy.** The hyperparameters were left in a sub-optimal state intentionally, as the focus of this project is to demonstrate how a model can be deployed and served via a RESTful API, rather than achieving high performance. This conscious decision allows users to explore how changes in model configuration impact accuracy and other evaluation metrics.

## Deployment

The API is hosted in https://elecomexp.pythonanywhere.com/, making it accessible from any device with an internet connection.

## Usage

1. Clone the repository.
2. Set up the virtual environment with **Python 3.10**.
3. Install the requiered dependencies
4. Run the Flask app locally or deploy it on PythonAnywhere.
