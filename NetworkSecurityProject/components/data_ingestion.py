from NetworkSecurityProject.exception.exception import NetworkSecurityException
from NetworkSecurityProject.logging.logger import logging

## configuration of the Data Ingestion Config
from NetworkSecurityProject.entity.config_entity import DataIngestionConfig
from NetworkSecurityProject.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()


MONGO_DB_URL=os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig): 
        try:
            self.data_ingestion_config = data_ingestion_config # using data_ingestion_config  defined in config_entity.yaml
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def export_collection_as_dataframe(self):
        """
        Read data from mongodb as a pandas dataframe
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            # when we read from mongo db by default it adds "_id" as a column hence we drop it
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            
            df.replace({"na":np.nan},inplace=True)
            return df
        
        except Exception as e:
            raise NetworkSecurityException
        

    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        ''' Exports read data from mongodb to a feature store
        '''
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            #creating a folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        ''' Splits dataframe from feature store into train and test for ML'''
        
        try:
            
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")


            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )


            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            

            os.makedirs(dir_path, exist_ok=True)
            

            logging.info(f"Exporting train and test file path.")
            

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            
            logging.info(f"Exported train and test file path.")
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def initiate_data_ingestion(self):

        ''' Initiates the entire Data ingestion Process incoporating the above defined functions and outputs dataingestionartifact'''

        try:
            dataframe = self.export_collection_as_dataframe() # reads data from mongodb and returns it as a pandas dataframe
            
            dataframe = self.export_data_into_feature_store(dataframe)
            
            self.split_data_as_train_test(dataframe)
            
            dataingestionartifact=DataIngestionArtifact(trained_file_path = self.data_ingestion_config.training_file_path, test_file_path = self.data_ingestion_config.testing_file_path)
            return dataingestionartifact # This is the output from Data Ingestion component

        except Exception as e:
            raise NetworkSecurityException