from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

PROJECT_DIR = "/opt/airflow/dags/Radar_project"

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),  # must be in the past
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="radar_mlops_pipeline",
    default_args=default_args,
    schedule_interval="@weekly",
    catchup=False,
    tags=["mlops", "dvc", "radar"],
) as dag:

    pull_data = BashOperator(
        task_id="pull_data",
        bash_command=f"""
        cd {PROJECT_DIR} &&
        dvc pull
        """,
        env={
            "DAGSHUB_USERNAME": os.environ.get("DAGSHUB_USERNAME"),
            "DAGSHUB_TOKEN": os.environ.get("DAGSHUB_TOKEN"),
        },
    )

    run_pipeline = BashOperator(
        task_id="run_dvc_pipeline",
        bash_command=f"""
        cd {PROJECT_DIR} &&
        dvc repro
        """,
        env={
            "DAGSHUB_USERNAME": os.environ.get("DAGSHUB_USERNAME"),
            "DAGSHUB_TOKEN": os.environ.get("DAGSHUB_TOKEN"),
            "MLFLOW_TRACKING_URI": "https://dagshub.com/shashwatsingh0511/radae_mlops.mlflow",
        },
    )

    push_results = BashOperator(
        task_id="push_results",
        bash_command=f"""
        cd {PROJECT_DIR} &&
        dvc push -j 1
        """,
        env={
            "DAGSHUB_USERNAME": os.environ.get("DAGSHUB_USERNAME"),
            "DAGSHUB_TOKEN": os.environ.get("DAGSHUB_TOKEN"),
        },
    )

    pull_data >> run_pipeline >> push_results
