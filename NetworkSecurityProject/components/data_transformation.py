
# For data_transformation component the input is Data validaton artifact and output is data transformation artifact.
#       Uses
#       1. Uses DataTransformationConfig defined in config_entity.py as input
#       2. Uses DataValidationArtifact and DataTransformationArtifact defined in artifact_entity.py file
#       3. Uses constants defined in constant/training_pipeline/__init__.py file

import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from NetworkSecurityProject.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from NetworkSecurityProject.entity.artifact_entity import (DataValidationArtifact, DataTransformationArtifact)

from NetworkSecurityProject.entity.config_entity import DataTransformationConfig
from NetworkSecurityProject.exception.exception import NetworkSecurityException 
from NetworkSecurityProject.logging.logger import logging
from NetworkSecurityProject.utils.main_utils.utils import save_numpy_array_data,save_object # # save_numpy_array_data, when we apply knn imputer we need to save the ouput as a pickle file which we defien in save_object function in utils.py

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                    data_transformation_config: DataTransformationConfig):

                    try:
                        self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
                        
                        self.data_transformation_config:DataTransformationConfig = data_transformation_config

                    except Exception as e:
                        raise NetworkSecurityException(e, sys)
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Initializes a KNNImputer object with parameters specified in training_pipeline.py
        and returns a Pipeline object with the KNNImputer as the first step.

        Args:
            cls: The class `DataTransformation`

        Returns:
            Pipeline: A Pipeline object with the KNNImputer step.
        """

        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            # Initialize the KNNImputer with specified parameters
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS) # it will take params defined in constant.training_pipeline.__init__.py

            logging.info(
                    f"Initialized KNNImputer with parameters: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            # Create and return the Pipeline with the imputer
            processor : Pipeline = Pipeline([("imputer",imputer)])
            return processor  
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:

        logging.info(
            "Entered initiate_data_transformation method of DataTransformation class"
        )

        train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
        
        test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)


        # Training data frame independent(input features) and dependent feature (target)
        input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN], axis = 1) # drop target column
        
        target_feature_train_df = train_df[TARGET_COLUMN]
        # As TARGET_COLUMN only contains +1, -1 hence for finary classification we replace -1 with 0
        target_feature_train_df = target_feature_train_df.replace(-1, 0)
      
        # Testing data frame independent(input features) and dependent feature (target)
        input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN], axis = 1) # drop target column
        
        target_feature_test_df = test_df[TARGET_COLUMN]
        target_feature_test_df = target_feature_test_df.replace(-1, 0)


        preprocessor = self.get_data_transformer_object()
        preprocessor_obj = preprocessor.fit(input_feature_train_df)
        transformed_input_train_features = preprocessor_obj.transform(input_feature_train_df)

        transformed_input_test_features = preprocessor_obj.transform(input_feature_test_df)

        train_arr = np.c_[transformed_input_train_features, np.array(target_feature_train_df)]
    
        test_arr = np.c_[transformed_input_test_features, np.array(target_feature_test_df)]


        #save numpy array data
        ## train
        save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array = train_arr,)
        ## test
        save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array = test_arr,)
       
        save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj,)

        # This creates a folder called final_model and saves the best data transformation preprocessor_obj object as a pickle file in it
        save_object("final_model/preprocessor.pkl", preprocessor_obj,)






        #preparing artifacts
        data_transformation_artifact = DataTransformationArtifact(
            transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,

            transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,

            transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
        )
        return data_transformation_artifact

    







