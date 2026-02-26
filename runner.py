import mlflow
import mlflow.sklearn
from joblib import load

from scripts import evaluate, process_data, train
from constants import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    MODEL_FILEPATH, ARTIFACTS_DIR,
)
from utils import load_params


def run():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        # Load and log all params
        data_params = load_params('process_data')
        train_params = load_params('train')

        mlflow.log_param('features', data_params['features'])
        mlflow.log_param('train_size', data_params.get('train_size', 'full'))
        mlflow.log_param('model_type', train_params.get('model_type', 'LogisticRegression'))

        model_params = {k: v for k, v in train_params.items() if k != 'model_type'}
        for k, v in model_params.items():
            mlflow.log_param(k, v)

        # Run pipeline steps
        process_data()
        train()
        metrics = evaluate()

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log artifacts
        mlflow.log_artifacts(ARTIFACTS_DIR)

        # Log model as MLflow model entity
        model = load(MODEL_FILEPATH)
        mlflow.sklearn.log_model(model, 'model')


if __name__ == '__main__':
    run()
