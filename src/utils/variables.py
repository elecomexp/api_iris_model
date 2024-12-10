"""
variables.py

Author: Lander Combarro Exposito
Created: December 04, 2024
Last Modified: December 04, 2024

Description
-----------
This module defines constants and configurations for the Iris Dataset Classification API,
including paths to data and models, and mappings for class labels to Iris flower species.
"""

# Relative paths
DATA_PATH = './data/iris.csv'
MODEL_PATH = './model/iris_model.pkl'

# Class mapping to Iris flower species names
CLASS_MAPPING = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}
