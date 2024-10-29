from NetworkSecurityProject.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact # Data Validation component uses generated dataa ingestion component
from NetworkSecurityProject.entity.config_entity import DataValidationConfig # **

from NetworkSecurityProject.exception.exception import NetworkSecurityException
from NetworkSecurityProject.logging.logger import logging

from scipy.stats import ks_2samp # library for checking data drift or not
import pandas as pd
import os, sys


from NetworkSecurityProject.constant.training_pipeline import SCHEMA_FILE_PATH # path of the schema.yaml
from NetworkSecurityProject.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation: # Input of the Data Validation Class/component is DataIngestionArtifact and output is DataValidationConfig
    def __init__(self, 
                data_ingestion_artifact:DataIngestionArtifact,
                data_validation_config:DataValidationConfig):
                try:
                    self.data_ingestion_artifact = data_ingestion_artifact
                    self.data_validation_config = data_validation_config
                    
                    self._schema_config = read_yaml_file(SCHEMA_FILE_PATH) # read_yaml_file present in main_util folder's util.py file ( contains All generic functions used in the project)

                except Exception as e:
                    raise NetworkSecurityException(e,sys)
    
    # This function is used only once hence defined as a static method
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    # 1. Validating number of columns    
    def validate_number_of_columns(self, dataframe:pd.DataFrame) -> bool:
        try:
            # number_of_columns = len(self._schema_config) # returns only 2 columns
            # Get the actual number of columns specified in the schema
            number_of_columns = len(self._schema_config.get("columns", [])) # returns 31 

            logging.info(f"Required number of columns:{number_of_columns}") # how many columns are there in schema

            logging.info(f"Data frame has columns:{len(dataframe.columns)}") # how many columns our dataframe has
            
            if len(dataframe.columns) == number_of_columns:  
                return True
            
            return False

        except Exception as e:
            raise NetworkSecurityException(e,sys)

    # 2. Validating presence of numerical columns by returning status and if any numerical columns missing in data then returning missing columns
    def validate_numerical_columns(self, dataframe:pd.DataFrame) -> bool:
        try:
            # Get the list of required numerical columns from schema if any returns a list of keys else a empty list 
            required_numerical_columns = self._schema_config.get("numerical_columns", [])
            
            # Get numerical columns present in the DataFrame as a list
            dataframe_numerical_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
            
            # Log the required and DataFrame's numerical columns
            logging.info(f"Required numerical columns: {required_numerical_columns}")

            logging.info(f"Numerical columns in DataFrame: {dataframe_numerical_columns}")
            
            # Check if all required numerical columns exist in the DataFrame
            missing_columns = [col for col in required_numerical_columns if col not in dataframe_numerical_columns]
            
            if not missing_columns:
                return True  # All required numerical columns are present
            else:
                logging.warning(f"Missing numerical columns in DataFrame: {missing_columns}")
                return False
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def detect_dataset_drift(self, base_df, current_df, threshold = 0.05) -> bool:
        try:
            status = True # by default, we set status as True for no drift
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                
                is_dataDriftFound = threshold > is_same_dist.pvalue # returns T or F
                if is_dataDriftFound:
                    status = False
                

                report.update({
                            column: {"p_value": float(is_same_dist.pvalue), "drift_status": is_dataDriftFound
                              }
                              })

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok = True)

            # write into drift_report_file_path
            write_yaml_file(file_path = drift_report_file_path, content = report)
            return status
            
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact: # Return type is DataValidationArtifact
        try:
            # Read data from train and test file paths
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Validate the number of columns in train and test DataFrames with columns present in schema
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status: # True
                error_message = f"Train dataframe does not contain all columns.\n" 
            
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status: # True
                error_message=f"Test dataframe does not contain all columns.\n" 

            # Validate that numerical columns in train and test DataFrames match the schema
            status = self.validate_numerical_columns(dataframe=train_dataframe)
            if not status:
                error_message = "Train dataframe is missing some required numerical columns."
                # logging.error(error_message)
                # raise ValueError(error_message)

            status = self.validate_numerical_columns(dataframe=test_dataframe)
            if not status:
                error_message = "Test dataframe is missing some required numerical columns."
                # logging.error(error_message)
                # raise ValueError(error_message)


           # Checking data drift between train and test DataFrames

            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            
            # Ensure the directory path exists for saving valid train and test files
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok = True)
            
            # Save the validated train and test DataFrames to specified paths
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True

            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            # Create a DataValidationArtifact to record validation details
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                
                invalid_train_file_path=None,
                invalid_test_file_path=None,
            
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            return data_validation_artifact #  validation_status, valid_train_file_path, valid_test_file_path, invalid_train_file_path, invalid_test_file_path, drift_report_file_path  

        except Exception as e:
            raise NetworkSecurityException(e,sys)