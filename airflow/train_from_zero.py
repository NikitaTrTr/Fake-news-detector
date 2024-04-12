# print('hello world')

from airflow.decorators import dag
from airflow.operators.python_operator import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from Parser immport parser
from Preprocessor import preprocessor
from Trainer import trainer

csv_path='data'
models_path='models'
parsing_period= ('2023-01-01', '2023-12-31')
models_set=('lr','rf','svm','cb')

dockerops_kwargs = { #X3X3X3X3
    "mount_tmp_dir": False,
    "mounts": [
        Mount(
            source="/home/ivan/studcamp/studcamp/airflow-ml/data", # Change to your path
            target="/opt/airflow/data/",
            type="bind",
        )
    ],
    "retries": 1,
    "api_version": "1.30",
    "docker_url": "tcp://docker-socket-proxy:2375",
    "network_mode": "bridge",
}

@dag("from_parse_to_model", start_date=days_ago(0), schedule="@daily", catchup=False)
def taskflow():
    load = PythonOperator(
        task_id="from_parse_to_model",
        python_callable=parser,
        op_kwargs={
            "path": csv_path,
            "period": parsing_period
        },
    )

    process = PythonOperator(
        task_id="from_parse_to_model",
        python_callable=preprocessor,
        op_kwargs={
            "path": csv_path,
        },
    )

    train = PythonOperator(
        task_id="from_parse_to_model",
        python_callable=trainer,
        op_kwargs={
            "path_from": csv_path,
            "path_to": models_path,
            "models": models_set
        },
    )
    push = DockerOperator(
        task_id="news_label",
        container_name="X3X3X3",
        image="X3X3X3",
        command=f"(git commit -m automatic) && (git push)",
        **dockerops_kwargs,
    )
    )

  load >> process >> train >> push


taskflow()

