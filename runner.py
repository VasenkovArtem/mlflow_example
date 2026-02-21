import mlflow
from constants import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from scripts import evaluate, process_data, train


if __name__ == '__main__':

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    process_data()
    train()
    evaluate()
