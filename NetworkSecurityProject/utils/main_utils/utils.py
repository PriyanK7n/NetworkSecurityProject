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
