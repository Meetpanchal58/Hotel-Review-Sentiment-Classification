from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
import pendulum
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 
from src.components.model_evaluation import ModelEvaluation

with DAG(
    "NLP_Complete_pipeline",
    default_args={"retries": 2},
    description="this is Complete NLP Pipeline",
    schedule="@weekly",# here you can test based on hour or mints but make sure here you container is up and running
    start_date=pendulum.datetime(2024, 4, 1, tz="UTC"),
    catchup=False,
    tags=["NLP","Deep Learning","Multi_Text_Classification"],
) as dag:
    
    dag.doc_md = __doc__


def Run_Data_ingestion_pipeline(**kwargs):
    data_ingestion = DataIngestion()
    df = data_ingestion.initiate_data_ingestion()
    kwargs['ti'].xcom_push(key='raw_df', value=df)


def Run_Data_transformation_pipeline(**kwargs):
    df = kwargs['ti'].xcom_pull(key='raw_df')
    data_transformation = DataTransformation()
    X_balanced, y_balanced = data_transformation.transform(df)
    kwargs['ti'].xcom_push(key='X_balanced', value=X_balanced)
    kwargs['ti'].xcom_push(key='y_balanced', value=y_balanced)


def Run_Model_trainer_pipeline(**kwargs):
    X_balanced = kwargs['ti'].xcom_pull(key='X_balanced')
    y_balanced = kwargs['ti'].xcom_pull(key='y_balanced')
    model_trainer = ModelTrainer()
    gru, X_test, y_test = model_trainer.train_model(X_balanced, y_balanced)
    kwargs['ti'].xcom_push(key='gru_model', value=gru)
    kwargs['ti'].xcom_push(key='X_test', value=X_test)
    kwargs['ti'].xcom_push(key='y_test', value=y_test)


def Run_Model_evaluation_pipeline(**kwargs):
    gru_model = kwargs['ti'].xcom_pull(key='gru_model')
    X_test = kwargs['ti'].xcom_pull(key='X_test')
    y_test = kwargs['ti'].xcom_pull(key='y_test')
    model_evaluation = ModelEvaluation()
    model_evaluation.evaluate_model(gru_model, X_test, y_test)


def push_to_AzureBlob():
        subprocess.run(["dvc", "commit"])
        subprocess.run(["dvc", "push", "-r", "MyStorage"])
      

data_ingestion_task = PythonOperator(
    task_id='run_data_ingestion_pipeline',
    python_callable=Run_Data_ingestion_pipeline,
    provide_context=True,  
    dag=dag,
)


data_transformation_task = PythonOperator(
    task_id='run_Data_transformation_pipeline',
    python_callable=Run_Data_transformation_pipeline,
    provide_context=True,  
    dag=dag,
)

model_trainer_task = PythonOperator(
    task_id='run_model_trainer_pipeline',
    python_callable=Run_Model_trainer_pipeline,
    provide_context=True,  
    dag=dag,
)

model_evaluation_task = PythonOperator(
    task_id='run_model_evaluation_pipeline',
    python_callable=Run_Model_evaluation_pipeline,
    provide_context=True,  
    dag=dag,
)


push_to_AzureBlob_task = PythonOperator(
    task_id='push_to_AzureBlob',
    python_callable=push_to_AzureBlob,
    dag=dag,
)


# Set task dependencies
data_ingestion_task >> data_transformation_task >> model_trainer_task >> model_evaluation_task >> push_to_AzureBlob_task