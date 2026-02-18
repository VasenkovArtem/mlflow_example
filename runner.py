import os
import mlflow

from scripts.process_data import process_data
from scripts.train import train
from scripts.evaluate import evaluate

MLFLOW_TRACKING_URI = "http://158.160.2.37:5000/"
EXPERIMENT_NAME = "homework_mavletova" 
RUN_NAME = "D_gradient_boosting_all_features"

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = RUN_NAME

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("pipeline", "adult_income")
        mlflow.set_tag("stage_order", "process_data->train->evaluate")

        process_data()
        train()
        evaluate()


if __name__ == "__main__":
    main()
