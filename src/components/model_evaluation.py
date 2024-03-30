import numpy as np
import dagshub
import mlflow
import mlflow.keras
import mlflow.sklearn
from mlflow import log_metric, log_param
from src.logger.logging import logging
from sklearn.metrics import confusion_matrix, classification_report
from src.exception.exception import CustomException


class ModelEvaluation:
    def __init__(self):
        pass

    def evaluate_model(self, gru, X_test, y_test):
        logging.info("Model evaluation started")
        
        try:
            dagshub.init(repo_owner='Meetpanchal58', repo_name='Hotel-Review-Sentiment-Classification', mlflow=True)  

            mlflow.set_tracking_uri("mlruns") 

            # Start MLflow run
            mlflow.start_run(run_name="Model Evaluation")

            y_pred = np.argmax(gru.predict(X_test),axis = 1)

            loss, accuracy = gru.evaluate(X_test, y_test)
            Report = classification_report(y_test, y_pred)
            Matrix = confusion_matrix(y_test, y_pred)

            log_metric("Test Loss", loss)  
            log_metric("Test Accuracy", accuracy)
            log_param(f"Confusion Matrix", Matrix.tolist())
            log_param(f"Classification Report", Report)
    
            logging.info("Model evaluation completed")
        
        except Exception as e:
            logging.exception("An error occurred during model evaluation")
            raise CustomException(e)
        
        finally:
            # End MLflow run
            mlflow.end_run()