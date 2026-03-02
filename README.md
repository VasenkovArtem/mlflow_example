# mlflow_example
Пример использования mlflow для трекинга экспериментов

Запуск пайплайна &mdash; `python3 runner.py`

# Эксп1

http://158.160.2.37:5000/#/experiments/27/runs/77e135e0b9c54ec18f197480851f51e7


train_size:

train_size_500 

Первый запуск, просто проверить

             precision    recall  f1-score   support

           0       0.80      0.97      0.88      7429
           1       0.73      0.21      0.33      2340

    accuracy                           0.79      9769
   macro avg       0.76      0.59      0.60      9769
weighted avg       0.78      0.79      0.75      9769

Чиловые результаты получили, для класса 1 явно видно что не справляется модель

train_size_2000

              precision    recall  f1-score   support

           0       0.80      0.98      0.88      7429
           1       0.77      0.21      0.33      2340

    accuracy                           0.80      9769
   macro avg       0.79      0.60      0.61      9769
weighted avg       0.79      0.80      0.75      9769

Лучше сильно не стало, просто precision качнули, но recall для 1 все еще отвратительный


Попробуем поменять модель:

exp_model_decision_tree


              precision    recall  f1-score   support

           0       0.86      0.85      0.85      7429
           1       0.54      0.56      0.55      2340

    accuracy                           0.78      9769
   macro avg       0.70      0.70      0.70      9769
weighted avg       0.78      0.78      0.78      9769

Ну тут сразу видно, что хорошие результаты уже пошли, да упал recall на 0, но кому не все равно

За такой буст на 1, что уже хотя бы лучше чем случайность, оно того стоит.

exp_model_random_forest

              precision    recall  f1-score   support

           0       0.84      0.92      0.88      7429
           1       0.63      0.45      0.53      2340

    accuracy                           0.81      9769
   macro avg       0.74      0.69      0.70      9769
weighted avg       0.79      0.81      0.79      9769


Случайный лес неплохо себя показал, но может его можно дотюнить?

exp_model_random_forest_max_depth_15



              precision    recall  f1-score   support

           0       0.85      0.89      0.87      7429
           1       0.57      0.49      0.53      2340

    accuracy                           0.79      9769
   macro avg       0.71      0.69      0.70      9769
weighted avg       0.78      0.79      0.78      9769

Кажется ключ все таки переход еще и с 1000-2000 данных

Ну и это 99% лучший скор, обычно всякие оптимизации оптимизаторов это знаки после запятой а не фундоментальный прорыв (если не DL, там всякое бывает)

exp_model_random_forest_max_depth_15_train_size_2000


             precision    recall  f1-score   support

           0       0.85      0.91      0.88      7429
           1       0.62      0.49      0.54      2340

    accuracy                           0.80      9769
   macro avg       0.73      0.70      0.71      9769
weighted avg       0.79      0.80      0.80      9769

Ну такое, улучшить метрики удалось, но вот пробить  0.5 recall для 1 не вышло, в целом все молодцы




http://158.160.2.37:5000/#/experiments/27/runs/4acaad90c26348cda3d6bbb811aad499/artifacts

roc-auc  ~ 0.79