import sys
import logging

import pandas as pd

sys.path.append('ml_pipeline/')
sys.path.append('ml_pipeline/parsers')
from airflow.decorators import dag, task
from Preprocessor import Preprocessor
from Trainer import Trainer
from LapshaParser import LapshaParser
from LentaParser import LentaParser
from PanoramaParser import PanoramaParser
import pendulum
import yaml

from datetime import datetime, date
from datetime import timedelta

with open('airflow/config_cold.yaml', 'r') as f:
    config = yaml.safe_load(f)


@dag(
    dag_id='train_from_zero',
    schedule="None",
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    default_args={
        "owner": "Nikita Tatarinov",
    },
    tags=['fake news']
)
def taskflow():
    @task(multiple_outputs=True)
    def parse_data():
        day_start= (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        day_current=date.today().strftime("%Y-%m-%d")
        lapsha_parser = LapshaParser()
        lenta_parser = LentaParser()
        panorama_parser = PanoramaParser()
        logging.info("Lenta is parsing")
        lenta_dict = {'dateFrom': day_start,
                      'dateTo': day_current}
        df_lenta = lenta_parser.get_articles(param_dict=lenta_dict, n_articles=config['lenta_articles'])
        logging.info("Panorama is parsing")
        df_panorama = panorama_parser.get_articles(day_start,
                                                   day_current,
                                                   config['panorama_articles'])
        logging.info("Parsing is done")
        return {'lenta': df_lenta, 'panorama': df_panorama}

    @task()
    def preprocess_data(lapsha, lenta, panorama):
        logging.info("Preprocessing data")
        preprocessor = Preprocessor()
        logging.info("1")
        df = preprocessor.preprocess(lapsha, panorama, lenta)
        logging.info("Preprocessing is done")
        return df

    @task()
    def continue_train(df):
        logging.info("Training model")
        trainer = Trainer(df)
        trainer.fine_tune_model(X=df[['text']], y=df['label'], model_name='catboost.pkl')
        logging.info("Training is done")
        return df
    @task.bash
    def push_update() -> str:
        return '(git commit -m dailymodel)&&(git push)'

    data = parse_data()
    processed_data = preprocess_data(data['lenta'], data['panorama'])
    continue_train(processed_data)

    taskflow()

