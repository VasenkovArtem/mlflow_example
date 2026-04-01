import os
from datetime import datetime

import mlflow

from constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from scripts import evaluate, process_data, train
from utils import load_params, setup_mlflow


def main():
    setup_mlflow()

    train_params = load_params('train')
    model_type = train_params.get('model_type', 'unknown_model')

    default_run_name = f'{model_type}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    run_name = os.getenv('RUN_NAME', default_run_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag('pipeline', 'adult_income_homework')

        optional_tags = {
            'study_slice': os.getenv('RUN_TAG_SLICE', ''),
            'study_value': os.getenv('RUN_TAG_VALUE', ''),
            'study_hypothesis': os.getenv('RUN_TAG_HYPOTHESIS', ''),
        }
        for key, value in optional_tags.items():
            if value:
                mlflow.set_tag(key, value)

        process_data()
        train()
        metrics = evaluate()

        if 'roc_auc' in metrics:
            mlflow.set_tag('best_metric_name', 'roc_auc')
            mlflow.set_tag('best_metric_value', str(metrics['roc_auc']))

        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is not None:
            run_url = (
                f"{MLFLOW_TRACKING_URI.rstrip('/')}"
                f"/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
            )
            print(f'\nMLflow run URL: {run_url}\n')


if __name__ == '__main__':
    main()
