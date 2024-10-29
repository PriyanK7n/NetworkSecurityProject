from NetworkSecurityProject.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from NetworkSecurityProject.components.data_ingestion import DataIngestion
from NetworkSecurityProject.components.data_validation import DataValidation
import sys
# from NetworkSecurityProject.entity.artifact_entity import DataIngestionArtifact
from NetworkSecurityProject.exception.exception import NetworkSecurityException
from NetworkSecurityProject.logging.logger import logging

if __name__ == "__main__":

    try:
        '''Data Ingestion Process'''
        trainingPipelineConfig = TrainingPipelineConfig()
        dataIngestionConfig= DataIngestionConfig(trainingPipelineConfig)
        data_ingestion_obj = DataIngestion(dataIngestionConfig)
        logging.info("Initating Data Ingestion Process")
        dataingestionartifact = data_ingestion_obj.initiate_data_ingestion()
        logging.info("Data Ingestion Process Completed")
        print(dataingestionartifact)

        '''Data Ingestion Process Finished'''

        data_validation_config=DataValidationConfig(trainingPipelineConfig)
        data_validation=DataValidation(dataingestionartifact, data_validation_config)
        
        logging.info("Initating Data Validation Process")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data Validation Process Completed")
        print(data_validation_artifact)
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)


