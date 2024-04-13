import sys
import logging

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
        lapsha_parser = LapshaParser()
        lenta_parser = LentaParser()
        panorama_parser = PanoramaParser()
        logging.info("Lenta is parsing")
        lenta_dict = {'dateFrom': config['lenta_parse_from'],
                      'dateTo': config['lenta_parse_to']}
        df_lenta = lenta_parser.get_articles(param_dict=lenta_dict, n_articles=config['lenta_articles'])
        logging.info("Lapsha is parsing")
        df_lapsha = lapsha_parser.get_articles(n_articles=config['lapsha_articles'])
        logging.info("Panorama is parsing")
        df_panorama = panorama_parser.get_articles(config['panorama_parse_from'],
                                                   config['panorama_parse_to'],
                                                   config['panorama_articles'])
        logging.info("Parsing is done")
        return {'lapsha': df_lapsha, 'lenta': df_lenta, 'panorama': df_panorama}

    @task()
    def preprocess_data(lapsha, lenta, panorama):
        logging.info("Preprocessing data")
        preprocessor = Preprocessor()
        logging.info("1")
        df = preprocessor.preprocess(lapsha, panorama, lenta)
        logging.info("Preprocessing is done")
        return df

    @task()
    def train_model(df):
        logging.info("Training model")
        trainer = Trainer(df)
        trainer.get_trained_model(X=df[['text']], y=df['label'], model_name=config['classifier'])
        logging.info("Training is done")
        return df

    data = parse_data()
    processed_data = preprocess_data(data['lapsha'], data['lenta'], data['panorama'])
    train_model(processed_data)


taskflow()
