from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "article_draft_ru.md"


def best_classification_model(metrics: dict) -> tuple[str, dict]:
    name = max(metrics, key=lambda key: metrics[key]["roc_auc"])
    return name, metrics[name]


def main() -> None:
    required = [
        REPORTS / "metrics.json",
        REPORTS / "level_model_metrics.json",
        REPORTS / "feature_importance.csv",
        REPORTS / "level_feature_importance.csv",
        REPORTS / "gmm_cluster_summary.csv",
        REPORTS / "ablation_study.csv",
        REPORTS / "cross_validation_summary.csv",
        REPORTS / "run_manifest.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing reports: {', '.join(str(path) for path in missing)}")

    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    level_metrics = json.loads((REPORTS / "level_model_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((REPORTS / "run_manifest.json").read_text(encoding="utf-8"))
    importance = pd.read_csv(REPORTS / "feature_importance.csv")
    level_importance = pd.read_csv(REPORTS / "level_feature_importance.csv")
    gmm = pd.read_csv(REPORTS / "gmm_cluster_summary.csv")
    ablation = pd.read_csv(REPORTS / "ablation_study.csv")
    cv = pd.read_csv(REPORTS / "cross_validation_summary.csv")
    data = pd.read_csv(ROOT / "data" / "processed" / "levels_results.csv")

    best_name, best = best_classification_model(metrics)
    level_best = level_metrics["metadata"]["best_model"]
    level_best_metrics = level_metrics[level_best]
    hard_cluster = gmm.sort_values("mean_difficulty", ascending=False).iloc[0]
    best_ablation = ablation.sort_values("roc_auc", ascending=False).iloc[0]
    best_cv = cv.sort_values("roc_auc_mean", ascending=False).iloc[0]

    top_model_features = ", ".join(importance.head(6)["feature"].astype(str).tolist())
    top_level_features = ", ".join(level_importance.head(6)["feature"].astype(str).tolist())
    agents = ", ".join(sorted(data["agent"].unique()))

    text = f"""# Прогнозирование сложности процедурно сгенерированных игровых уровней с помощью Game AI и вероятностного моделирования

## Аннотация

В работе предлагается воспроизводимый подход к оценке сложности процедурно сгенерированных игровых уровней. Вместо субъективной экспертной оценки сложность определяется как вероятность непрохождения уровня AI-агентом за ограниченное число шагов:

```text
difficulty = 1 - P(success)
```

Данные генерируются автоматически по фиксированным seed в собственной grid-world среде. Для каждого уровня извлекаются структурные признаки, запускаются несколько агентов, после чего обучаются модели машинного обучения и оценивается неопределенность прогноза.

## Постановка проблемы

При процедурной генерации уровней разработчик может получать большое количество вариантов карты. Ручная проверка сложности становится дорогой и плохо масштабируемой. Поэтому требуется метод, который автоматически оценивает, насколько уровень вероятно будет проходимым для агента с заданной стратегией.

## Гипотеза

Структурные признаки уровня, такие как длина кратчайшего пути, плотность препятствий, доступная площадь, тупики и ветвистость маршрутов, позволяют предсказывать вероятность успешного прохождения уровня AI-агентом.

## Данные

Датасет был получен не ручной загрузкой готового CSV, а воспроизводимой генерацией уровней:

- строк в датасете: {len(data)}
- уникальных уровней: {data["level_id"].nunique()}
- агенты: {agents}
- средняя сложность: {data["difficulty"].mean():.3f}
- SHA-256 датасета: `{manifest["data_sha256"]}`

Одна запись соответствует запуску конкретного агента на конкретном уровне и содержит признаки уровня, имя агента, результат прохождения, число шагов, reward и рассчитанную сложность.

## Методология

Pipeline состоит из следующих этапов:

1. Генерация grid-world уровней по seed.
2. Извлечение структурных признаков карты.
3. Запуск AI-агентов: random, greedy, A* и tabular Q-learning.
4. Расчет целевой переменной `success`.
5. Обучение моделей Logistic Regression, SVM, Random Forest и Bayesian Logistic Regression.
6. EM-кластеризация уровней через Gaussian Mixture Model.
7. Ablation study для проверки вклада групп признаков.
8. Group cross-validation по `level_id`, чтобы избежать утечки одного уровня между train и test.
9. Calibration curve для проверки вероятностного прогноза.
10. Оценка важности признаков и построение графиков.

## Результаты классификации

Лучшая модель для предсказания успешного прохождения: `{best_name}`.

- ROC-AUC: {best["roc_auc"]:.3f}
- F1: {best["f1"]:.3f}
- Accuracy: {best["accuracy"]:.3f}
- Brier score: {best["brier"]:.3f}

Наиболее важные признаки в общей модели: {top_model_features}.

## Анализ сложности уровня

Дополнительно была обучена модель, которая предсказывает среднюю сложность уровня только по структурным признакам, без использования признака `agent`.

- лучшая модель: `{level_best}`
- R2: {level_best_metrics["r2"]:.3f}
- MAE: {level_best_metrics["mae"]:.3f}
- RMSE: {level_best_metrics["rmse"]:.3f}

Главные структурные признаки: {top_level_features}.

Этот результат важен для проверки гипотезы, потому что показывает, что сложность связана не только с типом агента, но и со структурой самого уровня.

## EM-кластеризация

Gaussian Mixture Model использовалась как EM-подход для группировки уровней без заранее заданных классов сложности. Самый сложный кластер: `{int(hard_cluster["em_cluster"])}` со средней difficulty={hard_cluster["mean_difficulty"]:.3f}.

## Ablation study

Лучший вариант в ablation study: `{best_ablation["experiment"]}` с ROC-AUC={best_ablation["roc_auc"]:.3f}. Это подтверждает, что совместное использование структурных признаков и информации об агенте дает наиболее точный прогноз.

## Group cross-validation

Для проверки устойчивости использовался GroupKFold по `level_id`. Лучший результат показала модель `{best_cv["model"]}`:

- mean ROC-AUC: {best_cv["roc_auc_mean"]:.3f}
- std ROC-AUC: {best_cv["roc_auc_std"]:.3f}
- mean F1: {best_cv["f1_mean"]:.3f}

## Оригинальность

Оригинальность проекта состоит не в простом применении одной Python-библиотеки, а в разработке полного исследовательского pipeline:

- собственная генерация уровней;
- самостоятельное определение метрики сложности;
- реализация A* и Q-learning агентов;
- извлечение структурных признаков уровня;
- сравнение frequentist, Bayesian и EM-подходов;
- оценка неопределенности;
- возможность повторить эксперимент на новых seed или новом CSV той же структуры.

## Воспроизводимость

Для повторения эксперимента сохраняются конфиг, версии библиотек, хэш датасета и список артефактов в `reports/run_manifest.json`. Полный запуск выполняется командой:

```powershell
python .\\run_pipeline.py run-all
```

Если изменить seed, размеры уровней или плотность препятствий в `configs/default.json`, pipeline заново сгенерирует данные, обучит модели и обновит отчеты.

## Ограничения

Текущая версия использует абстрактную grid-world среду, а не полноценный коммерческий игровой движок. Это делает эксперимент контролируемым и воспроизводимым, но в дальнейшей работе метод можно расширить на MiniGrid, Unity ML-Agents или реальные уровни конкретной игры.

## Вывод

Полученные результаты показывают, что сложность процедурно сгенерированного уровня можно формализовать как вероятностную величину и предсказывать по структурным признакам карты. Такой подход может использоваться как инструмент предварительной балансировки уровней в game development.
"""
    OUT.write_text(text, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
