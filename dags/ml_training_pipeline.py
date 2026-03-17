from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
from airflow.sensors.python import PythonSensor
import os
import json

default_args = {
    'owner': 'orest',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

F1_THRESHOLD = 0.55

def check_accuracy(**kwargs):
    
    metrics_path = '/opt/airflow/project/models/metrics.json'
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        f1_score = metrics.get('f1', 0)
        print(f"Current F1 Score: {f1_score}")
        
        if f1_score >= F1_THRESHOLD:
            return 'register_model' 
        else:
            return 'stop_pipeline'   
    except Exception as e:
        print(f"Error reading metrics: {e}")
        return 'stop_pipeline'

with DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Continuous Training Pipeline for Telco Churn',
    schedule_interval=None, 
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'lab5'],
) as dag:

    check_data = PythonSensor(
        task_id='check_data_availability',
        python_callable=lambda: os.path.exists('/opt/airflow/project/data/raw/telco_churn.csv'),
        poke_interval=5,
        timeout=60,
    )

    run_dvc_pipeline = BashOperator(
        task_id='run_dvc_pipeline',
        bash_command='cd /opt/airflow/project && dvc repro',
    )

    evaluate_model = BranchPythonOperator(
        task_id='evaluate_model',
        python_callable=check_accuracy,
    )

    register_model = BashOperator(
        task_id='register_model',
        bash_command="""
        cd /opt/airflow/project && python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
with mlflow.start_run(run_name='airflow_pipeline'):
    mlflow.log_artifact('models/model.pkl', artifact_path='model')
    run_id = mlflow.active_run().info.run_id
    model_uri = f'runs:/{run_id}/model'
    # Реєструємо і переводимо в Staging
    mv = mlflow.register_model(model_uri, 'Telco_Churn_Airflow')
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name='Telco_Churn_Airflow', version=mv.version, stage='Staging')
print('Model successfully registered in Staging!')
"
        """
    )

    stop_pipeline = EmptyOperator(
        task_id='stop_pipeline',
    )

    check_data >> run_dvc_pipeline >> evaluate_model >> [register_model, stop_pipeline]