import os, sys
from NetworkSecurityProject.exception.exception import NetworkSecurityException 
from NetworkSecurityProject.logging.logger import logging


from NetworkSecurityProject.entity.config_entity import DataTransformationConfig
from NetworkSecurityProject.utils.main_utils.utils import save_numpy_array_data,save_object # # save_numpy_array_data, when we apply knn imputer we need to save the ouput as a pickle file which we defien in save_object function in utils.py

from NetworkSecurityProject.components.data_ingestion import DataIngestion
from NetworkSecurityProject.components.data_validation import DataValidation
from NetworkSecurityProject.components.data_transformation import DataTransformation
from NetworkSecurityProject.components.model_trainer import ModelTrainer
from NetworkSecurityProject.constant.training_pipeline import TRAINING_BUCKET_NAME

from NetworkSecurityProject.entity.config_entity import (
    TrainingPipelineConfig, 
    DataIngestionConfig, 
    DataValidationConfig, 
    DataTransformationConfig, 
    ModelTrainerConfig
    )

from NetworkSecurityProject.entity.artifact_entity import (
    DataIngestionArtifact, 
    DataValidationArtifact, 
    DataTransformationArtifact, 
    ModelTrainerArtifact
    )

from NetworkSecurityProject.cloud.s3_syncer import S3Sync # imports custom aws cli function for syncing local directory folder with aws s3

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync_obj = S3Sync() # initialize  # imports custom aws cli function for syncing local directory folder with aws s3

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config =  DataIngestionConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("Initating Data Ingestion Process")

            data_ingestion_obj = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion_obj.initiate_data_ingestion()
            logging.info (f"Data Ingestion Process Completed and Data Ingestion Artifact created, {data_ingestion_artifact}")
            
            return data_ingestion_artifact

            
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config = self.training_pipeline_config)

            data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
            logging.info("Initating Data Validation Process")
            data_validation_artifact=data_validation.initiate_data_validation()   
            logging.info("Data Validation Process Completed")            
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config = self.training_pipeline_config)

            logging.info("Initating Data Transformation Process")
            data_transformation = DataTransformation(data_validation_artifact = data_validation_artifact, data_transformation_config = data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            # print(data_transformation_artifact)
            logging.info("Data Transformation Process Complted and Data Transformation Artifact created")
            return data_transformation_artifact
            
       
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def start_model_trainer(self, data_transformation_artifact : DataTransformationArtifact) -> ModelTrainerArtifact:

        try:
            logging.info("Initating Model Training Process")
            print("Model Trainer uses data_transformation_artifact and ModelTrainerConfig as Input")

            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(training_pipeline_config = self.training_pipeline_config)

            model_trainer = ModelTrainer( 
                                        data_transformation_artifact = data_transformation_artifact,
                                        model_trainer_config = self.model_trainer_config)

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            logging.info("Model Trainer Artifact Created and Model Training Process Completed")
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # local artifact folder is going to aws s3 bucket with timestamps recorded
    def sync_artifact_dir_to_s3(self):
        # TRAINING_BUCKET_NAME present in NetworkSecurityProject/constant/training_pipeline/__init__.py
        # TRAINING_BUCKET_NAME = "networksecuritybucket"

        try:
            # url to uploaded artifacts folder to s3 with timestamps recorded
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync_obj.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir, aws_bucket_url = aws_bucket_url)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # local final model folder is going to aws s3 bucket
    def sync_saved_model_dir_to_s3(self):
        try:
            # url to uploaded models to s3
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync_obj.sync_folder_from_s3(folder = self.training_pipeline_config.model_dir, aws_bucket_url=aws_bucket_url)
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def run_training_pipeline(self):
        
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)

            data_transformation_artifact = self.start_data_transformation(data_validation_artifact = data_validation_artifact)
            
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact = data_transformation_artifact)

            # syncing with aws s3
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)





