import mlflow

from constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from scripts import evaluate, process_data, train
from utils import load_params


def _run_name() -> str:
    process_params = load_params('process_data')
    train_params = load_params('train')
    train_size = process_params.get('train_size') or 'all'
    model_type = train_params.get('model_type', 'unknown')
    key_param = None
    if model_type == 'logistic_regression':
        key_param = ('C', train_params.get('C'))
    elif model_type == 'decision_tree':
        key_param = ('max_depth', train_params.get('max_depth'))
    elif model_type == 'random_forest':
        key_param = ('n_estimators', train_params.get('n_estimators'))
    elif model_type == 'xgboost':
        key_param = ('max_depth', train_params.get('max_depth'))
    if key_param is not None:
        name, val = key_param
        if val is not None:
            return f"train_size={train_size}_model={model_type}_{name}={val}"
    return f"train_size={train_size}_model={model_type}"


if __name__ == '__main__':
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=_run_name()):
        process_data()
        train()
        evaluate()
