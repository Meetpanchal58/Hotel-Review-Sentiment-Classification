from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 
from src.components.model_evaluation import ModelEvaluation
from src.logger.logging import logging
from src.exception.exception import CustomException


class Training_Pipline:
    def __init__(self):
        pass

    def training_run(self):
        try:
            logging.info("Training Pipeline run is started")
            # Data ingestion
            data_ingestion = DataIngestion()
            df = data_ingestion.initiate_data_ingestion()
 
            # Data transformation
            data_transformation = DataTransformation()
            X_balanced, y_balanced = data_transformation.transform(df)
            
            # Model training
            model_trainer = ModelTrainer()
            X_test, y_test = model_trainer.train_model(X_balanced, y_balanced)
             
            model_evaluation = ModelEvaluation()
            model_evaluation.evaluate_model(X_test, y_test)
        
            logging.info("Training Pipline run is completed successfully")
        
        except CustomException as e:
            logging.error(f"An error occurred during Training process: {e}")
            raise

if __name__ == "__main__":
    pipeline = Training_Pipline()
    pipeline.training_run()


