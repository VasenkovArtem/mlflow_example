from scripts import evaluate, process_data, train
from utils import init_mlflow

import mlflow
import os


if __name__ == '__main__':
    init_mlflow()

    # Allow marking type of experiment via env var for easier analysis in MLflow UI
    experiment_type = os.getenv('EXPERIMENT_TYPE')

    with mlflow.start_run():
        if experiment_type:
            mlflow.set_tag('experiment_type', experiment_type)

        mlflow.set_tag('pipeline', 'adult-census-income')
        process_data()
        train()
        evaluate()
