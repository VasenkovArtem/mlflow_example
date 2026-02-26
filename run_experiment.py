import yaml


def write_yaml(filepath, params):
    with open(filepath, 'w') as f:
        yaml.dump({'params': params}, f, default_flow_style=False)


def run_single():
    from runner import run
    run()


DEFAULT_DATA_PARAMS = {
    'features': [
        'race', 'sex', 'native.country', 'occupation', 'education', 'capital.gain',
    ],
    'train_size': 6000,
}

DEFAULT_TRAIN_PARAMS = {
    'model_type': 'LogisticRegression',
    'penalty': 'l2',
    'C': 0.9,
    'solver': 'lbfgs',
    'max_iter': 1000,
}

EVALUATE_PARAMS = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision'],
}


def run_experiments():
    write_yaml('params/evaluate.yaml', EVALUATE_PARAMS)

    print('Experiments: varying train_size')
    for train_size in [2000, 4000, 6000, 8000]:
        print(f'train_size={train_size}')
        data_params = {**DEFAULT_DATA_PARAMS, 'train_size': train_size}
        write_yaml('params/process_data.yaml', data_params)
        write_yaml('params/train.yaml', DEFAULT_TRAIN_PARAMS)
        run_single()

    print('Experiments: varying model_type')
    model_configs = {
        'LogisticRegression': {'penalty': 'l2', 'C': 0.9, 'solver': 'lbfgs', 'max_iter': 1000},
        'DecisionTree': {'max_depth': 10},
        'RandomForest': {'n_estimators': 100, 'max_depth': 10},
        'GradientBoosting': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
    }
    for model_type, model_params in model_configs.items():
        print(f'model_type={model_type}')
        write_yaml('params/process_data.yaml', DEFAULT_DATA_PARAMS)
        train_params = {'model_type': model_type, **model_params}
        write_yaml('params/train.yaml', train_params)
        run_single()

    print('Experiments: varying C for LogisticRegression')
    for C in [0.01, 0.1, 1.0, 10.0]:
        print(f'C={C}')
        write_yaml('params/process_data.yaml', DEFAULT_DATA_PARAMS)
        train_params = {
            'model_type': 'LogisticRegression',
            'penalty': 'l2',
            'C': C,
            'solver': 'lbfgs',
            'max_iter': 1000,
        }
        write_yaml('params/train.yaml', train_params)
        run_single()


if __name__ == '__main__':
    run_experiments()
