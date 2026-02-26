# Отчёт по домашнему заданию: Трекинг экспериментов с MLflow

## Описание экспериментальной серии

Проведена серия из 14 экспериментов с варьированием параметров в четырёх разрезах. Все запуски находятся в одном эксперименте MLflow (имя из `constants.EXPERIMENT_NAME`). Таблицы и графики можно получить в MLflow UI: выберите эксперимент → отметьте нужные runs → **Compare**; в Compare можно сортировать по параметрам и строить графики зависимости метрик от выбранного параметра.

---

## Разрез 1: Размер тренировочного датасета

**Гипотеза:** Увеличение размера тренировочного датасета улучшает качество модели.

**Параметр:** `data_train_size` (размер тренировочного датасета)

**Сетка значений:** 2000, 5000, 10000, 15000

**Фиксированные параметры:**
- Модель: Logistic Regression (penalty=l2, C=0.9, solver=lbfgs)
- Фичи: age, education, capital.gain, hours.per.week, race, sex, occupation

### Результаты

| train_size | accuracy | precision | recall | f1_score | roc_auc | pr_auc |
|------------|----------|-----------|--------|----------|---------|--------|
| 2000       | 0.791    | 0.664     | 0.254  | 0.368    | 0.801   | 0.594   |
| 5000       | 0.791    | 0.664     | 0.262  | 0.375    | 0.802   | 0.595   |
| 10000      | 0.792    | 0.666     | 0.267  | 0.381    | 0.802   | 0.594   |
| 15000      | 0.792    | 0.660     | 0.268  | 0.382    | 0.802   | 0.594   |

В MLflow UI: фильтр по параметру `data_train_size`, затем Compare выбранных runs. График: в Compare выберите по оси X параметр `data_train_size`, по оси Y — `roc_auc`.

### Выводы

Гипотеза частично подтверждается: с ростом размера обучающей выборки с 2000 до 5000–10000 метрики немного улучшаются (roc_auc с 0.801 до 0.802), далее при 15000 стабилизируются. Для логистической регрессии на данном наборе признаков выигрыш от добавления данных после 5–10 тыс. объектов невелик.

---

## Разрез 2: Тип модели

**Гипотеза:** Ансамблевые модели (Random Forest, Gradient Boosting) превосходят простые модели по ROC-AUC.

**Параметр:** `model_type`

**Значения:** logistic_regression, decision_tree, random_forest, gradient_boosting

**Фиксированные параметры:**
- train_size: 10000
- Фичи: age, education, capital.gain, hours.per.week, race, sex, occupation

### Результаты

| model_type        | roc_auc | accuracy | f1_score |
|-------------------|---------|----------|----------|
| logistic_regression | ...   | ...      | ...      |
| decision_tree     | ...     | ...      | ...      |
| random_forest     | ...     | ...      | ...      |
| gradient_boosting | ...     | ...      | ...      |

### Выводы

...

---

## Разрез 3: Гиперпараметры Logistic Regression

**Гипотеза:** Увеличение коэффициента регуляризации C и смена solver влияют на качество.

**Параметры:** `model_C`, `model_solver`

**Сетка:** (C=0.1, lbfgs), (C=1.0, lbfgs), (C=10.0, saga)

### Результаты

| C    | solver | roc_auc |
|------|--------|---------|
| 0.1  | lbfgs  | 0.802   |
| 1.0  | lbfgs  | 0.802   |
| 10.0 | saga   | 0.411   |

Для C=10.0 и solver=saga обучение не успело сойтись (max_iter=1000), что привело к деградации качества.

### Выводы

Для lbfgs изменение C в диапазоне 0.1–1.0 почти не влияет на roc_auc. Переход на saga при C=10.0 без увеличения max_iter ухудшает результат; для saga нужны большие max_iter или другая подготовка данных.

---

## Разрез 4: Набор признаков

**Гипотеза:** Расширенный набор признаков улучшает качество предсказания.

**Параметр:** `data_features` (количество и состав признаков)

**Варианты:**
- minimal: age, education, capital.gain, race, sex
- base: + hours.per.week, occupation
- extended: + workclass, marital.status, relationship, native.country, capital.loss

### Результаты

| features_set | roc_auc |
|-------------|---------|
| minimal     | ...     |
| base        | ...     |
| extended    | ...     |

### Выводы

...

---

## Лучший запуск по ROC-AUC

**Ссылка на run:** [Run с наилучшим ROC-AUC (Gradient Boosting)](http://158.160.2.37:5000/#/experiments/31/runs/8a21d0fb985b4c41b998f8f531f99260)

В MLflow UI: эксперимент `homework_student` (или ваш EXPERIMENT_NAME) → сортировка по метрике `roc_auc` по убыванию — лучший run будет первым. Ссылку можно скопировать из адресной строки при открытии этого run.

**Параметры лучшего запуска:**
- data_train_size: 10000
- data_features: base (age, education, capital.gain, hours.per.week, race, sex, occupation)
- model_type: gradient_boosting
- model_n_estimators: 100, model_max_depth: 5, model_learning_rate: 0.1
- Метрики: roc_auc ≈ 0.884, accuracy ≈ 0.846, f1_score ≈ 0.625

Эти параметры записаны в `params/best_params.yaml`; для воспроизведения скопируйте их в `params/process_data.yaml`, `params/train.yaml`, `params/evaluate.yaml` и выполните `python runner.py`.
