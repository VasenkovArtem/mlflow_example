from scripts import evaluate, process_data, train
from utils import load_params
import mlflow


if __name__ == '__main__':
    mlflow.set_tracking_uri('http://158.160.2.37:5000/')
    mlflow.set_experiment('homework_ershov')

    params = load_params('common')
    run_name = params['run_name']

    with mlflow.start_run(run_name=run_name):
        process_data()
        train()
        evaluate()

