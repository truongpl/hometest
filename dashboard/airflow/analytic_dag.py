import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pendulum
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago


# This airflow will execute tasks:
# 1. Get analytic predictions and groundtruth
# 2. Calculate metrics
# 3. Insert metrics to analytic db

default_args = {
    'owner': 'Truong Le',
    'depends_on_past': False,
    "retries": 0,
    "retry_delay": pendulum.duration(seconds=20),
    'start_date': days_ago(1),
    'template_searchpath':'./'
}

dag = DAG(dag_id="analytic_pipeline",
    default_args=default_args,
    description="Analytic Pipeline")

get_predictor_data_task = BashOperator(
    task_id='get_predictor_task',
    bash_command='python get_predict_data.py',
    dag=dag
)

calculate_metrics_task = BashOperator(
    task_id='calculate_metrics_task',
    bash_command='python calculate_metrics_task.py',
    dag=dag
)

insert_analytic_task = BashOperator(
    task_id='insert_analytic_task',
    bash_command='python insert_analytic_task.py',
    dag=dag
)

get_predictor_data_task >> calculate_metrics_task >> insert_analytic_task
