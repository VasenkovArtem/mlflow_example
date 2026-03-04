import mlflow

MLFLOW_TRACKING_URI = "http://158.160.2.37:5000/"
EXPERIMENT_NAME = "homework_Borevsky"

def launch_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_mlflow(**kwargs):
    for key, value in kwargs.items():
        notation, text = key.split('_', 1)
        if notation == 'param':
            mlflow.log_param(text, value)
        elif notation == 'metric':
            mlflow.log_metric(text, value)
        else:
            raise ValueError(f"Unknown notation: {notation}. Use 'param_' or 'metric_'")