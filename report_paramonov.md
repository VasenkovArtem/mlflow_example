# Анализ экспериментов коллеги

**Эксперимент MLflow:** `homework_paramonov` (experiment_id=29)  
**Адрес MLflow:** http://158.160.2.37:5000/#/experiments/29  
**Автор анализа:** Timur Lidzhiev

---

## Обзор

Коллега провёл 16 запусков (из них 2 без метрик — неуспешные). Анализ основан на 14 успешных запусках. Использовались 4 типа моделей: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier и GradientBoostingClassifier. Эксперименты отличались по типу модели, набору признаков и гиперпараметрам.

---

## Сводная таблица успешных запусков

| # | Run Name | Модель | Признаки (кратко) | cat/num | max_depth | n_est | lr | ROC-AUC | Accuracy | F1 |
|---|----------|--------|-------------------|---------|-----------|-------|-----|---------|----------|-----|
| 1 | wistful-cat-218 | LogisticRegression | race, sex, native.country, occupation, education, capital.gain | 5/1 | — | — | — | 0.7127 | 0.7978 | 0.3137 |
| 2 | caring-squid-852 | LogisticRegression | race, sex, native.country, occupation, education, capital.gain | 5/1 | — | — | — | 0.7254 | 0.7982 | 0.3275 |
| 3 | salty-worm-258 | LogisticRegression | workclass, education, marital.status, occupation, relationship | 5/0 | — | — | — | 0.7065 | 0.7617 | 0.0026 |
| 4 | dazzling-moth-741 | LogisticRegression | age, education.num, marital.status, capital.gain, capital.loss | 1/4 | — | — | — | 0.8207 | 0.8079 | 0.4624 |
| 5 | smiling-steed-747 | DecisionTree | age, sex, education.num, capital.gain, hours.per.week | 1/4 | 5 | — | — | 0.8399 | 0.8322 | 0.5749 |
| 6 | grandiose-shrew-87 | DecisionTree | age, sex, education.num, capital.gain, hours.per.week | 1/4 | 5 | — | — | 0.8399 | 0.8322 | 0.5749 |
| 7 | skillful-koi-431 | DecisionTree | age, education.num, capital.gain, capital.loss, hours.per.week | 0/5 | 5 | — | — | 0.8276 | 0.8369 | 0.5492 |
| 8 | fortunate-bass-706 | DecisionTree | workclass, marital.status, occupation, capital.gain, capital.loss | 3/2 | 10 | — | — | 0.8942 | 0.8549 | 0.6467 |
| 9 | serious-grub-24 | RandomForest | age, education.num, marital.status, relationship, hours.per.week | 2/3 | 10 | 100 | — | 0.8788 | 0.8303 | 0.5942 |
| 10 | languid-moose-404 | RandomForest | sex, education, capital.gain, capital.loss, hours.per.week | 2/3 | — | 200 | — | 0.8429 | 0.8409 | 0.5765 |
| 11 | selective-gnu-738 | RandomForest | age, workclass, education, ... (10 признаков) | 6/4 | 15 | 50 | — | 0.9154 | 0.8634 | 0.6735 |
| 12 | redolent-ant-349 | RandomForest | age, workclass, education, ... (10 признаков) | 6/4 | 15 | 50 | — | 0.9158 | 0.8638 | 0.6740 |
| 13 | orderly-grouse-64 | GradientBoosting | age, workclass, education.num, occupation, capital.gain | 2/3 | 3 | 100 | 0.1 | 0.8573 | 0.8282 | 0.5412 |
| 14 | honorable-wolf-438 | GradientBoosting | sex, marital.status, relationship, capital.gain, hours.per.week | 3/2 | 5 | 200 | 0.05 | 0.8700 | 0.8194 | 0.5518 |

---

## Разрез 1: Влияние типа модели

### Описание

- **Гипотеза:** ансамблевые модели (RandomForest, GradientBoosting) превосходят одиночные модели (LogisticRegression, DecisionTree) по ROC-AUC.
- **Исследуемый параметр:** `model_type`
- **Значения:** LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier
- **Примечание:** сравниваются лучшие запуски каждого типа модели.

### Результаты

| Модель | Лучший ROC-AUC | Accuracy | F1 | Run Name |
|--------|---------------|----------|-----|----------|
| LogisticRegression | 0.8207 | 0.8079 | 0.4624 | dazzling-moth-741 |
| DecisionTreeClassifier | 0.8942 | 0.8549 | 0.6467 | fortunate-bass-706 |
| RandomForestClassifier | **0.9158** | **0.8638** | **0.6740** | redolent-ant-349 |
| GradientBoostingClassifier | 0.8700 | 0.8194 | 0.5518 | honorable-wolf-438 |

### Выводы

Гипотеза **частично подтвердилась**. RandomForest показал лучший результат (ROC-AUC = 0.916), значительно превосходя LogisticRegression (+0.095). Однако GradientBoosting (0.870) оказался слабее DecisionTree (0.894). Вероятная причина — GradientBoosting тестировался с менее информативным набором признаков (5 фичей) и консервативными гиперпараметрами (lr=0.05–0.1, depth=3–5), тогда как лучший DecisionTree использовал depth=10 и включал сильные предикторы (marital.status, occupation).

---

## Разрез 2: Влияние набора признаков

### Описание

- **Гипотеза:** расширение набора признаков (особенно добавление информативных категориальных переменных вроде marital.status, occupation, relationship) улучшает качество модели.
- **Исследуемый параметр:** набор признаков (`features`) и их количество
- **Фиксированные параметры:** модель — RandomForestClassifier, train_size=26048

### Результаты

| Признаки | Кол-во | cat/num | ROC-AUC | F1 | Run Name |
|----------|--------|---------|---------|-----|----------|
| sex, education, capital.gain, capital.loss, hours.per.week | 5 | 2/3 | 0.8429 | 0.5765 | languid-moose-404 |
| age, education.num, marital.status, relationship, hours.per.week | 5 | 2/3 | 0.8788 | 0.5942 | serious-grub-24 |
| age, workclass, education, marital.status, occupation, relationship, sex, capital.gain, capital.loss, hours.per.week | 10 | 6/4 | **0.9158** | **0.6740** | redolent-ant-349 |

### Выводы

Гипотеза **полностью подтвердилась**. Расширение набора признаков с 5 до 10 привело к росту ROC-AUC с 0.843 до 0.916 (+0.073). Также видно, что не только количество, но и состав признаков критически важен: набор из 5 фичей с marital.status и relationship (0.879) значительно лучше набора из 5 фичей без них (0.843). Переменные marital.status, relationship и occupation — сильнейшие предикторы дохода.

---

## Разрез 3: Влияние гиперпараметров модели (max_depth)

### Описание

- **Гипотеза:** увеличение max_depth дерева решений улучшает ROC-AUC за счёт более сложных разбиений, но может привести к переобучению.
- **Исследуемый параметр:** `max_depth` для DecisionTreeClassifier
- **Значения:** 5, 10
- **Фиксированные параметры:** модель — DecisionTreeClassifier, train_size=26048

### Результаты

| max_depth | Признаки | ROC-AUC | Accuracy | F1 | Run Name |
|-----------|----------|---------|----------|-----|----------|
| 5 | age, sex, education.num, capital.gain, hours.per.week | 0.8399 | 0.8322 | 0.5749 | smiling-steed-747 |
| 5 | age, education.num, capital.gain, capital.loss, hours.per.week | 0.8276 | 0.8369 | 0.5492 | skillful-koi-431 |
| 10 | workclass, marital.status, occupation, capital.gain, capital.loss | **0.8942** | **0.8549** | **0.6467** | fortunate-bass-706 |

### Выводы

Увеличение max_depth с 5 до 10 дало значительный прирост ROC-AUC: с 0.828–0.840 до 0.894. Однако здесь одновременно менялся и набор признаков, поэтому эффект нельзя однозначно отнести только к глубине дерева. Тем не менее, рост на +0.054 до +0.067 по ROC-AUC свидетельствует о том, что при depth=5 модель была недостаточно сложной (underfitting). Признаков переобучения при depth=10 не наблюдается.

---

## Лучший запуск

**Лучший результат по ROC-AUC: 0.9158**

- **Модель:** RandomForestClassifier (max_depth=15, n_estimators=50)
- **Признаки:** age, workclass, education, marital.status, occupation, relationship, sex, capital.gain, capital.loss, hours.per.week (10 фичей)
- **Run Name:** redolent-ant-349
- **Run ID:** `20f02e57ebaa4a73ac88a10f1bb30176`
- **Ссылка:** [http://158.160.2.37:5000/#/experiments/29/runs/20f02e57ebaa4a73ac88a10f1bb30176](http://158.160.2.37:5000/#/experiments/29/runs/20f02e57ebaa4a73ac88a10f1bb30176)

---

## Общие наблюдения

1. Коллега методично исследовал разные аспекты: тип модели, набор признаков, глубину деревьев, learning rate для бустинга.
2. Наиболее значимый фактор качества — **набор признаков**: включение marital.status, occupation и relationship даёт наибольший прирост ROC-AUC.
3. RandomForest с широким набором признаков (10 штук) и достаточной глубиной (15) показал лучший результат, опережая даже GradientBoosting (который тестировался с более бедным набором фичей).
4. Эксперимент с LogisticRegression + L1 + только категориальными признаками (salty-worm-258) провалился (ROC-AUC=0.706, F1≈0) — вероятно, из-за неудачного кодирования категорий и сильной регуляризации (C=0.1).
