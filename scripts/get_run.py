import mlflow
from mlflow.tracking import MlflowClient

def get_run():
    mlflow.set_tracking_uri('http://158.160.2.37:5000/')
    client = MlflowClient()

    run = client.create_run(
        experiment_id=19,
        tags={'mlflow.runName': 'random_forest_n_es_500'}
    )
    return mlflow, run