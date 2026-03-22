import os
import mlflow
from scripts import evaluate, process_data, train


if __name__ == '__main__':

    mlflow.set_tracking_uri("http://158.160.2.37:5000")
    mlflow.set_experiment("homework_student18")
    
    run_name = os.environ.get('MLFLOW_RUN_NAME', 'matt_run')
    
    with mlflow.start_run(run_name=run_name):
        process_data()
        train()
        evaluate()
