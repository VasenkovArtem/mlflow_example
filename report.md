# Отчёт: MLflow — базовый уровень (homework_Nikulin)

**Experiment:** `homework_Nikulin` — [MLflow UI](http://158.160.2.37:5000/).

Таблицы ниже построены по экспорту run из MLflow (сравнение запусков в UI даёт те же метрики).

---

## Разрез 1: размер тренировочной выборки (`train_size`)

**Гипотеза:** увеличение объёма обучающей выборки повышает ROC-AUC до насыщения.

**Параметр:** `train_size` ∈ {2000, 4000, 6000, 8000, 10000}.  
**Зафиксировано:** логистическая регрессия (L2, C=0.9, lbfgs), 6 признаков.

| Run name | train_size | ROC-AUC | PR-AUC | F1 |
|----------|------------|---------|--------|-----|
| slice_train_size_2000 | 2000 | 0.7233 | 0.5186 | 0.3312 |
| slice_train_size_4000 | 4000 | 0.7245 | 0.5191 | 0.3291 |
| slice_train_size_6000 | 6000 | 0.7273 | 0.5220 | 0.3286 |
| slice_train_size_8000 | 8000 | **0.7278** | **0.5221** | 0.3288 |
| slice_train_size_10000 | 10000 | 0.7260 | 0.5197 | 0.3291 |

**Выводы:** ROC-AUC растёт от 2000 до 8000 примеров, затем на 10000 слегка падает (шум/конкретное разбиение). Гипотеза подтверждается частично: есть рост, но плато и небольшой спад на 10000.

**Лучший ROC-AUC в разрезе:** [slice_train_size_8000](http://158.160.2.37:5000/#/experiments/39/runs/8c4fdd47dc5b42dca2dc9149621ca12a) (Run ID `8c4fdd47dc5b42dca2dc9149621ca12a`).

---

## Разрез 2: тип модели

**Гипотеза:** ансамбли и деревья дают выше ROC-AUC, чем линейная модель на тех же данных.

**Параметр:** `model_type`. **Зафиксировано:** train_size=8000, 6 признаков.

| Run name | model_type | ROC-AUC | PR-AUC | F1 |
|----------|------------|---------|--------|-----|
| slice_model_logistic_regression | logistic_regression | 0.7278 | 0.5221 | 0.3288 |
| slice_model_decision_tree | decision_tree | 0.8135 | 0.6277 | 0.4621 |
| slice_model_random_forest | random_forest | 0.8289 | 0.6712 | 0.5157 |
| slice_model_gradient_boosting | gradient_boosting | **0.8437** | **0.6929** | **0.5370** |

**Выводы:** гипотеза подтверждается: от LR к GB монотонный рост ROC-AUC; лучший — градиентный бустинг.

**Лучший ROC-AUC в разрезе:** [slice_model_gradient_boosting](http://158.160.2.37:5000/#/experiments/39/runs/9eeb7219a5974630a99a165b3a4877fe).

---

## Разрез 3: гиперпараметры LR (коэффициент `C`)

**Гипотеза:** экстремальные значения `C` хуже средних; оптимум в «середине» диапазона.

**Параметр:** `C` ∈ {0.01, 0.1, 1.0, 10.0}. **Зафиксировано:** LR, train_size=8000, 6 признаков, max_iter=2000.

| Run name | C | ROC-AUC | PR-AUC | F1 |
|----------|---|---------|--------|-----|
| slice_lr_C_0.01 | 0.01 | 0.7258 | 0.5203 | 0.3282 |
| slice_lr_C_0.1 | 0.1 | 0.7275 | 0.5220 | 0.3287 |
| slice_lr_C_1.0 | 1.0 | 0.7278 | 0.5221 | 0.3288 |
| slice_lr_C_10.0 | 10.0 | **0.7278** | 0.5221 | 0.3287 |

**Выводы:** при C от 0.1 до 10 метрики почти не меняются; при C=0.01 ROC-AUC чуть ниже. Ярко выраженного «провала» на краях нет — гипотеза слабо подтверждается (есть лёгкий штраф только при очень сильной регуляризации).

**Лучший ROC-AUC в разрезе:** [slice_lr_C_10.0](http://158.160.2.37:5000/#/experiments/39/runs/c467c1b0e8314829afced8cd98a10ebf) (на доли выше, чем C=1.0; на практике можно считать C≈1–10 эквивалентными).

---

## Дополнительно: набор признаков (2 run)

| Run | признаки | ROC-AUC |
|-----|----------|---------|
| slice_features_minimal | 4 (education, occupation, capital.gain, age) | 0.7378 |
| slice_features_full | 8 (+ race, sex, native.country, hours.per.week) | **0.8017** |

Расширение набора признаков заметно улучшает LR при том же train_size=8000.

---

## Лучший run по ROC-AUC (вся серия)

| Метрика | Значение |
|---------|----------|
| ROC-AUC | **0.8437** |
| PR-AUC | 0.6929 |
| F1 | 0.5370 |

**Ссылка:** [slice_model_gradient_boosting](http://158.160.2.37:5000/#/experiments/39/runs/9eeb7219a5974630a99a165b3a4877fe)  
(Run ID: `9eeb7219a5974630a99a165b3a4877fe`)

**Воспроизведение** (конфиги совпадают с этим run):

- [`params/process_data_best.yaml`](params/process_data_best.yaml) — 6 фичей, train_size 8000  
- [`params/train_best.yaml`](params/train_best.yaml) — `gradient_boosting`, n_estimators=100, max_depth=4, learning_rate=0.1  
- [`params/evaluate_best.yaml`](params/evaluate_best.yaml)

```bash
cp params/process_data_best.yaml params/process_data.yaml
cp params/train_best.yaml params/train.yaml
cp params/evaluate_best.yaml params/evaluate.yaml
python3 runner.py
```

*Если в вашем MLflow другой `experiment_id` в URL (не `39`), замените фрагмент `/experiments/39/` в ссылках на актуальный из адресной строки UI.*
