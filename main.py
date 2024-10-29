from NetworkSecurityProject.entity.config_entity import TrainingPipelineConfig
from NetworkSecurityProject.entity.config_entity import DataIngestionConfig
from NetworkSecurityProject.components.data_ingestion import DataIngestion

# from NetworkSecurityProject.entity.artifact_entity import DataIngestionArtifact
from NetworkSecurityProject.exception.exception import NetworkSecurityException
from NetworkSecurityProject.logging.logger import logging

if __name__ == "__main__":

    '''Data Ingestion Process Started'''
    try:
        trainingPipelineConfig = TrainingPipelineConfig()
        dataIngestionConfig= DataIngestionConfig(trainingPipelineConfig)
        data_ingestion_obj = DataIngestion(dataIngestionConfig)
        logging.info("Initating Data Ingestion Process")
        dataingestionartifact = data_ingestion_obj.initiate_data_ingestion()
        print(dataingestionartifact)
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
    '''Data Ingestion Process Finished'''


