# The Model trainer component takes ModelTrainer Config file and data transformation artifact as input and outputs Model trainer artifact
from NetworkSecurityProject.constant.training_pipeline import (MODEL_TRAINER_DIR_NAME, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME, MODEL_TRAINER_EXPECTED_SCORE, MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD, TRAINING_BUCKET_NAME)

# This file takes in Model Trainer ralated constants defined in NetworkSecurityProject/constant/training_pipeline/
#       Uses
#       1. Uses Model Trainer ralated constants defined in NetworkSecurityProject/constant/training_pipeline/
#       2. Uses ModelTrainerConfig class defined in the NetworkSecurityProject/entity/config_entity file 
# and DataTransformationArtifact defined in artifact_entity.py file
#       3. Uses constants defined in constant/training_pipeline/__init__.py file

import os
import sys

from NetworkSecurityProject.exception.exception import NetworkSecurityException 

from NetworkSecurityProject.logging.logger import logging

from NetworkSecurityProject.entity.artifact_entity import (DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact)

from NetworkSecurityProject.entity.config_entity import (TrainingPipelineConfig, DataTransformationConfig, ModelTrainerConfig, ModelTrainerConfig)


# common utility functions import from main_utils folder/package
from NetworkSecurityProject.utils.main_utils.utils import load_numpy_array_data, save_object, load_object

# ml utility functions import from ml_utils folder/package
from NetworkSecurityProject.utils.ml_utils.metric.classification_metric import get_classification_score
from NetworkSecurityProject.utils.ml_utils.model.estimator import NetworkModel

# Tried ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,)

from NetworkSecurityProject.utils.main_utils.utils import evaluate_models
import mlflow


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact # gives acces to     transformed_object_file_path, transformed_train_file_path, transformed_test_file_path

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def track_mlflow(self, best_model, classification_metric):
        
        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score

            # logging in MLFlow
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.sklearn.log_model(best_model, "model")


    # In this function we apply multiple ML algorithims
    def train_model(self, X_train, y_train, X_test, y_test):
        
        models = {  
                "RandomForest" : RandomForestClassifier(verbose = 1),

                "DecisionTree" : DecisionTreeClassifier(),
                
                "GradientBoosting" : GradientBoostingClassifier(verbose = 1),
                
                "LogisticRegression" : LogisticRegression(verbose = 1),
    
                "AdaBoost":AdaBoostClassifier()
                 }

        # Hyper parameter tuning using grid search CV
        # params dictionary is passed in evaluate_models function
        
        # Hyperparameter tuning configurations for each model
        params = {
            "DecisionTree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                # 'splitter': ['best', 'random'],
                # 'max_features': ['sqrt', 'log2'],
            },
            "RandomForest": {
                # 'criterion': ['gini', 'entropy', 'log_loss'],
                # 'max_features': ['sqrt', 'log2', None],
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "GradientBoosting": {
                # 'loss': ['log_loss', 'exponential'],
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                # 'criterion': ['squared_error', 'friedman_mse'],
                # 'max_features': ['auto', 'sqrt', 'log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "LogisticRegression": {},
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
                }

        # evaluate_models is present in NetworkSecurityProject/utils/main_utils/utils.py file, # This function unets and evaluates ML models to optimize performance on the phishing dataset and returns each model's r2 performance score

        model_report : dict = evaluate_models(
                        X_train = X_train, 
                        y_train = y_train, 
                        
                        X_test = X_test, 
                        y_test = y_test, 

                        models = models,
                        param = params)
        

        ## To get best model score from model_report dictionary 
        best_model_score = max(sorted(model_report.values())) # max(sort of r2 scores of each models)

        ## To get best model name from model_report dictionary 
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name] # this will store best model found's model calling function

        # Training performance metrics
        y_train_pred = best_model.predict(X_train)
        # get_classification_score is present in NetworkSecurityProject/utils/ml_utils/metric/classification_metric.py file and returns classification_train_metric which is an object of dataclass  ClassificationMetricArtifact containing best model's f1 score, precision_score and recall_score in here training scores
        classification_train_metric = get_classification_score(y_true = y_train, y_pred = y_train_pred) 
        #####################################
        # ML Flow Experiments Tracking:
        self.track_mlflow(best_model, classification_train_metric)
        #####################################


        # Testing performance metrics
        y_test_pred = best_model.predict(X_test)
        # Testing performance scores of best model 
        classification_test_metric = get_classification_score(y_true = y_test, y_pred = y_test_pred)
        #####################################
        # ML Flow Experiments Tracking:
        self.track_mlflow(best_model, classification_test_metric)
        #####################################



        # Save model and preprocessor
        # Using NetworkModel class which contains model and preprocessor to transform data before passing to model
        # load_object is used to load preprocessor pickle file and is defined in NetworkSecurityProject/utils/main_utils/utils.py file
        preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        # Using NetworkModel class which contains model and preprocessor to transform data before passing to model and is defined in NetworkSecurityProject/utils/ml_utils/model/estimator.py
        Network_Model = NetworkModel(preprocessor = preprocessor, model = best_model)
        
        # save_object is used to store object as a pickle file by Pushing Best Model and is defined in NetworkSecurityProject/utils/main_utils/utils.py file
        save_object(self.model_trainer_config.trained_model_file_path, obj = Network_Model)
        save_object("final_model/model.pkl",best_model)

        # Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
                                trained_model_file_path = self.model_trainer_config.trained_model_file_path, 
                                train_metric_artifact = classification_train_metric,
                                test_metric_artifact = classification_test_metric
                                                    )
        
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact



    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path 

            # load training and test numpy arrays as np.array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            

            # Split training arrays
            X_train = train_arr[:, :-1] # Features/Independent variables (all columns except "Result") for model training
            y_train = train_arr[:, -1] # Labels/Dependent variable ("Result" column) to evaluate training performance

            # Split test arrays
            X_test = test_arr[:, :-1]  # Features/Independent variables for model testing
            y_test = test_arr[:, -1]   # Labels/Dependent variable to assess model performance on test data


            # Train model and return ModelTrainer class artifact
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)