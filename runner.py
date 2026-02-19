import os
import sys
import yaml

from scripts import evaluate, process_data, train
import mlflow


def get_project_root():
    """Get project root (directory containing params/). Works for both local and Docker (/app)."""
    return os.path.dirname(os.path.abspath(__file__))


def load_experiments_config():
    """Load my_experiments.yaml and return experiment names -> config paths."""
    root = get_project_root()
    config_path = os.path.join(root, 'params', 'my_experiments.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config, root


if __name__ == '__main__':
    mlflow.set_tracking_uri("http://158.160.2.37:5000")
    mlflow.set_experiment("homework_student25")

    experiments_config, project_root = load_experiments_config()

    # Filter: env EXPERIMENT=exp_name or 1st arg, e.g. python runner.py exp_C_5
    filter_exp = os.environ.get('EXPERIMENT') or (sys.argv[1] if len(sys.argv) > 1 else None)
    if filter_exp and filter_exp not in experiments_config:
        print(f"Unknown experiment: {filter_exp}. Available: {list(experiments_config.keys())}")
        sys.exit(1)
    experiments = [filter_exp] if filter_exp else experiments_config.keys()

    for exp_name in experiments:
        paths = experiments_config[exp_name]
        # Resolve paths relative to project root
        process_data_path = os.path.join(project_root, paths['process_data'])
        train_path = os.path.join(project_root, paths['train'])
        evaluate_path = os.path.join(project_root, paths['evaluate'])

        run_name = os.environ.get('experiment_name', exp_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param('experiment', exp_name)
            process_data(params_filepath=process_data_path)
            train(params_filepath=train_path)
            evaluate(params_filepath=evaluate_path)
