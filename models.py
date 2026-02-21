from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def instantiate_model(model_name: str, params: dict):
    if model_name == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(**params)
    elif model_name == 'RandomForestClassifier':
        return RandomForestClassifier(**params)
    elif model_name == 'GradientBoostingClassifier':
        return GradientBoostingClassifier(**params)
    else:
        return LogisticRegression(**params)