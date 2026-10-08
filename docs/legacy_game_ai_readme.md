# Game AI Level Complexity (Legacy Prototype)

Проект оценивает сложность процедурно сгенерированных игровых уровней как вероятность непрохождения уровня AI-агентом.

```text
difficulty = 1 - P(success)
```

Это не демонстрация одной Python-библиотеки, а воспроизводимый исследовательский pipeline:

1. генерация уровней по `seed`;
2. извлечение структурных признаков уровня;
3. запуск нескольких AI-агентов: random, greedy, A* и Q-learning;
4. формирование датасета;
5. сравнение ML-моделей;
6. EM-кластеризация уровней;
7. приближенная байесовская оценка неопределенности;
8. group cross-validation по `level_id`;
9. calibration curve для вероятностного прогноза;
10. объяснение результата через важность признаков.

## Быстрый запуск

```powershell
python .\run_pipeline.py run-all
```

Предсказание для нового CSV с колонками `level_id`, `seed`, признаками уровня и `agent`:

```powershell
python .\run_pipeline.py predict --input .\data\processed\sample_new_levels.csv
```

Пример feature-only CSV уже лежит в `data/processed/sample_new_levels.csv`.

Результаты сохраняются в:

- `data/processed/levels_results.csv`
- `reports/metrics.json`
- `reports/predictions.csv`
- `reports/feature_importance.csv`
- `reports/gmm_cluster_summary.csv`
- `reports/ablation_study.csv`
- `reports/cross_validation_summary.csv`
- `reports/calibration_curve.csv`
- `reports/level_model_metrics.json`
- `reports/level_difficulty_predictions.csv`
- `reports/level_feature_importance.csv`
- `reports/run_manifest.json`
- `reports/plots/`
- `reports/plots/level_examples.png`
- `docs/defense_brief_ru.md`
- `docs/dataset_schema_ru.md`

## Тесты

```powershell
python -m unittest discover -s tests
```

## Приложение

```powershell
streamlit run .\app.py
```

В приложении можно загрузить другой CSV той же структуры или использовать уже сгенерированный датасет.

## Черновик статьи

После запуска pipeline можно собрать черновик академического отчета:

```powershell
python .\scripts\build_article.py
```

Файл сохраняется в `reports/article_draft_ru.md`.

## Что реализовано самостоятельно

- генерация grid-world уровней по seed;
- расчет кратчайшего пути и структурных признаков;
- A* агент;
- greedy агент;
- tabular Q-learning агент;
- правило сложности `difficulty = 1 - P(success)`;
- валидация датасета перед обучением.
- предсказание для нового CSV без целевых колонок.
- group cross-validation по `level_id`, чтобы избежать leakage между train и test.
- calibration curve для проверки качества вероятностного прогноза.
- визуализация easy/medium/hard уровней, восстановленных по seed.
- manifest запуска с конфигом, хэшем датасета и версиями библиотек.

Библиотеки `scikit-learn`, `pandas` и `matplotlib` используются для обучения моделей, метрик и визуализации, но не заменяют исследовательскую постановку и Game AI pipeline.
