'''
In practice, data ingestion can be more complex and may involve additional steps such as data validation, 
error handling, and integration with data storage systems. 
The above code provides a basic framework for reading a CSV file into a DataFrame and performing some initial checks on the data.
 Depending on the specific requirements of your project, you may need to customize the data ingestion process further to
handle different data formats, perform data cleaning, or integrate with other data sources.
'''
import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "train.csv")
    test_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "test.csv")
    raw_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "data.csv")
    source_data_path: str = os.path.join(PROJECT_ROOT, "notebook", "data", "stud.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    #from src.components.data_transformation import DataTransformation
    #from src.components.model_trainer import ModelTrainer

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    #data_transformation = DataTransformation()
    #train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    #modeltrainer = ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
