import os

import mlflow

from constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from scripts import evaluate, process_data, train


if __name__ == '__main__':
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_name = os.environ.get('MLFLOW_RUN_NAME')
    tags = {}
    slice_tag = os.environ.get('MLFLOW_EXPERIMENT_SLICE')
    if slice_tag:
        tags['experiment_slice'] = slice_tag

    with mlflow.start_run(run_name=run_name, tags=tags or None):
        process_data()
        train()
        evaluate()
