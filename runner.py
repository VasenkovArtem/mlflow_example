import mlflow

from scripts import evaluate, process_data, train
from constants import EXPERIMENT_NAME, MLFLOW_TRACKING_URI


if __name__ == '__main__':
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name='Log_dataset_2;LogReg;train=8000'):
        process_data()
        train(start_new_run=False)
        evaluate(start_new_run=False)
