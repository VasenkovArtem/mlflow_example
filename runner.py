import argparse
from scripts import evaluate, process_data, train, optimize
import mlflow
from setup import launch_mlflow


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLflow Homework Pipeline' )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['logreg', 'optuna'],
        default='logreg',
        help='Режим работы: logreg (стандартный пайплайн) или optuna (автооптимизация)'
    )
    
    args = parser.parse_args()
    launch_mlflow()
    if args.mode == 'logreg':
        with mlflow.start_run(run_name="logreg_pipeline"):
            process_data()
            train()
            evaluate()
    elif args.mode == 'optuna':
        with mlflow.start_run(run_name="optuna_data_prep"):
            process_data()
        optimize()