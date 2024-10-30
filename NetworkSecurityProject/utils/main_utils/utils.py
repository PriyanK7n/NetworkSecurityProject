# All generic functions used in the project are defined here 
import yaml
from NetworkSecurityProject.exception.exception import NetworkSecurityException
from NetworkSecurityProject.logging.logger import logging

import os,sys
import numpy as np
# import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# function for reading yaml files from provided path and returning a dictionary(.yaml) 
def read_yaml_file(file_path: str) -> dict:
        '''
        # Returns this
        {
        "columns": [
            {"having_IP_Address": "int64"},
            {"URL_Length": "int64"},
            ...
            {"Result": "int64"}
        ],
        "numerical_columns": [
            "having_IP_Address",
            "URL_Length",
            ...
            "Result"
        ]
        }
        '''

        try:
            with open(file_path, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e


# function for writing yaml files from provided path and returning a dictionary(.yaml) 
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# function to Save numpy array data to file
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    

# function to save knn imputer output as a pickle file    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# function defined to load pickle files 
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")

        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:

    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# This function unets and evaluates ML models to optimize performance on the phishing dataset.
# The models are evaluated using r2_score and classification metrics like f1_score, precision_score, recall_score
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
        
        '''
        Evaluates the performance of different machine learning models.

        **Arguments:**

        X_train (pandas.DataFrame): Training data features.
        y_train (pandas.Series): Training data target labels.
        X_test (pandas.DataFrame): Testing data features.
        y_test (pandas.Series): Testing data target labels.
        
        models (dict): Dictionary containing model objects as values and names as keys.
        **Returns:**
        dict: Dictionary containing test scores (R-squared) for each evaluated model.
        
        **Raises:**
        NetworkSecurityException: If an exception occurs during the evaluation process.

        '''

        try:

            report = {}
            
            # Loop through each model
            for i in range(len(list(models))):

                # gets each models' model object: ['RandomForestClassifier(verbose = 1)', 'DecisionTreeClassifier()', etc]
                model = list(models.values())[i]
                
                # param is a hyperparameter dictionary containing each model as key and values are a list of hyperparams
                para = param[list(models.keys())[i]] # get a list of corresponding hyperprams for corresponding models from params dictionary

                # Conducts hyperparameter tuning (GridSearchCV with 3-fold CV)
                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                # sets best hyperameter found for each model
                model.set_params(**gs.best_params_)

                # Train each model only on training dataset
                model.fit(X_train,y_train)

                # Make predictions on training and testing sets
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate R-squared scores
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                
                # Store test dataset r2 performance score of each model
                report[list(models.keys())[i]] = test_model_score
            
            return report

        except Exception as e:
            raise NetworkSecurityException(e, sys)