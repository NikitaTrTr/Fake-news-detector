# print('hello world')

from airflow.decorators import dag, task
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from Parser immport parser
from Preprocessor import preprocessor
from Trainer import warm_trainer

csv_path='data'
models_path='models'
parsing_short_period= ('2023-01-01', '2023-12-31')
models_set=('lr','rf','svm','cb')

@dag("from_parse_to_model", start_date=days_ago(1), schedule="@daily",  catchup=False)
def taskflow():
    load = PythonOperator(
        task_id="from_parse_to_model",
        python_callable=parser,
        op_kwargs={
            "path": csv_path,
            "period": parsing_short_period
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
        python_callable=warm_trainer,
        op_kwargs={
            "path_from": csv_path,
            "path_to": models_path
        },
    )

    @task.bash
    def push_task() -> str:
        return '(git commit -m dailymodel)&&()git push'

    push=push_task()

    load >> process >> train >> push


    taskflow()

