import mlflow

from constants import MLFLOW_TRACKING_URI, EXPERIMENT_NAME
from scripts import evaluate, process_data, train


if __name__ == '__main__':
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        process_data()
        train()
        evaluate()
