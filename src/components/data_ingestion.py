import io
import os
from azure.storage.blob import BlobServiceClient
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import CustomException
from src.utils.utils import save_csv
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_file_path=os.path.join('artifacts','hotel_reviews.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # AWS S3 details
            connection_string = os.environ.get("AZURE_CONNECTION_STRING")
            Container_name = 'database'
            blob_name = 'hote_reviews'

            # Load dataset from Azure Blob
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service_client.get_container_client(Container_name)
            blob_client = container_client.get_blob_client(blob_name)
            blob_data = blob_client.download_blob().readall()

            df = pd.read_csv(io.BytesIO(blob_data))
            df.drop(df.columns[0], axis=1, inplace=True)
            logging.info("Dataset loaded successfully from Azure blob")

            save_csv(
                file_path=self.data_ingestion_config.raw_file_path,
                data=df             
            )
            
            logging.info("Dataset saved to local artifacts folder")

            return df

        except Exception as e:
            logging.exception("An error occurred during data ingestion")
            raise CustomException(e)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()