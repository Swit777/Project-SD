from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .bayesian import BayesianLogisticLaplace
from .dataset import validate_dataset
from .features import FEATURE_COLUMNS
from .visualization import plot_level_examples


CATEGORICAL_COLUMNS = ["agent"]
MODEL_INPUT_COLUMNS = FEATURE_COLUMNS + CATEGORICAL_COLUMNS


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> ColumnTransformer:
    numeric_columns = FEATURE_COLUMNS if numeric_columns is None else numeric_columns
    categorical_columns = CATEGORICAL_COLUMNS if categorical_columns is None else categorical_columns
    transformers = []
    if numeric_columns:
        transformers.append(("num", StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(("cat", _one_hot_encoder(), categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped)),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, clipped)) if len(set(y_true)) > 1 else 0.5
    return result


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classifier_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", SVC(kernel="rbf", C=1.2, probability=True, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=220,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            names.extend(columns)
        elif name == "cat":
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out(columns).tolist())
            else:
                categories = transformer.categories_[0]
                names.extend([f"agent_{value}" for value in categories])
    return names


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _plot_metrics(metrics: dict[str, dict[str, float]], output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(metrics).T.sort_values("roc_auc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    frame[["roc_auc", "f1", "accuracy"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model comparison")
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "model_comparison.png", dpi=160)
    plt.close(fig)


def _plot_ablation(ablation: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(ablation["experiment"], ablation["roc_auc"], color=["#00a6c8", "#48a978", "#ffc457"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Ablation study: feature groups")
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "ablation_study.png", dpi=160)
    plt.close(fig)


def _plot_difficulty_by_density(data: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    grouped = (
        data.groupby(["requested_obstacle_density", "agent"], as_index=False)["difficulty"]
        .mean()
        .sort_values("requested_obstacle_density")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for agent, part in grouped.groupby("agent"):
        ax.plot(part["requested_obstacle_density"], part["difficulty"], marker="o", label=agent)
    ax.set_title("Average difficulty by obstacle density")
    ax.set_xlabel("requested obstacle density")
    ax.set_ylabel("difficulty")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "difficulty_by_density.png", dpi=160)
    plt.close(fig)


def _plot_uncertainty(predictions: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    grouped = predictions.groupby("agent", as_index=False)["uncertainty_width"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(grouped["agent"], grouped["uncertainty_width"], color="#75a4ff")
    ax.set_title("Bayesian uncertainty by agent")
    ax.set_ylabel("mean 90% interval width")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "uncertainty_by_agent.png", dpi=160)
    plt.close(fig)


def _plot_level_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(predictions["actual_difficulty"], predictions["predicted_difficulty"], alpha=0.65, s=18)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#444", linewidth=1)
    ax.set_title("Level difficulty: predicted vs actual")
    ax.set_xlabel("actual difficulty")
    ax.set_ylabel("predicted difficulty")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "level_predicted_vs_actual.png", dpi=160)
    plt.close(fig)


def _plot_calibration(calibration: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#444", linewidth=1, label="perfect")
    ax.plot(calibration["mean_predicted_probability"], calibration["fraction_of_positives"], marker="o", label="model")
    ax.set_title("Calibration curve")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "calibration_curve.png", dpi=160)
    plt.close(fig)


def _plot_cv_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(summary["model"], summary["roc_auc_mean"], yerr=summary["roc_auc_std"], color="#48a978", capsize=4)
    ax.set_ylim(0, 1.0)
    ax.set_title("Group cross-validation")
    ax.set_ylabel("ROC-AUC mean +/- std")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "cross_validation.png", dpi=160)
    plt.close(fig)


def _train_bayesian(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    random_state: int,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    preprocessor = make_preprocessor()
    x_train_p = preprocessor.fit_transform(x_train)
    x_test_p = preprocessor.transform(x_test)

    model = BayesianLogisticLaplace(prior_precision=1.0, random_state=random_state).fit(x_train_p, y_train)
    probs = model.predict_proba(x_test_p)[:, 1]
    preds = (probs >= 0.5).astype(int)
    low, high = model.probability_interval(x_test_p)
    uncertainty = pd.DataFrame(
        {
            "probability_low_05": low,
            "probability_high_95": high,
            "uncertainty_width": high - low,
        }
    )
    return _metrics(y_test, preds, probs), uncertainty, probs


def _gmm_summary(train_data: pd.DataFrame, output_dir: Path, random_state: int) -> pd.DataFrame:
    scaler = StandardScaler()
    values = scaler.fit_transform(train_data[FEATURE_COLUMNS])
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=random_state)
    clusters = gmm.fit_predict(values)
    summary_data = train_data.copy()
    summary_data["em_cluster"] = clusters
    summary = (
        summary_data.groupby("em_cluster")
        .agg(
            rows=("success", "size"),
            mean_difficulty=("difficulty", "mean"),
            mean_success=("success_rate", "mean"),
            mean_density=("actual_obstacle_density", "mean"),
            mean_path=("shortest_path_length", "mean"),
        )
        .reset_index()
        .sort_values("mean_difficulty")
    )
    summary.to_csv(output_dir / "gmm_cluster_summary.csv", index=False, encoding="utf-8")
    return summary


def _ablation_study(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_state: int,
    output_dir: Path,
) -> pd.DataFrame:
    experiments = {
        "agent_only": ([], CATEGORICAL_COLUMNS),
        "structure_only": (FEATURE_COLUMNS, []),
        "full_pipeline": (FEATURE_COLUMNS, CATEGORICAL_COLUMNS),
    }
    rows = []
    y = data["success"].astype(int).to_numpy()

    for name, (numeric_cols, categorical_cols) in experiments.items():
        cols = numeric_cols + categorical_cols
        model = Pipeline(
            [
                ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
                ),
            ]
        )
        model.fit(data.iloc[train_idx][cols], y[train_idx])
        prob = model.predict_proba(data.iloc[test_idx][cols])[:, 1]
        pred = model.predict(data.iloc[test_idx][cols])
        rows.append({"experiment": name, **_metrics(y[test_idx], pred, prob)})

    frame = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    frame.to_csv(output_dir / "ablation_study.csv", index=False, encoding="utf-8")
    _plot_ablation(frame, output_dir)
    return frame


def _cross_validate_models(
    data: pd.DataFrame,
    output_dir: Path,
    random_state: int,
    cv_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = data[MODEL_INPUT_COLUMNS]
    y = data["success"].astype(int).to_numpy()
    groups = data["level_id"].to_numpy()
    unique_groups = data["level_id"].nunique()
    n_splits = max(2, min(int(cv_folds), int(unique_groups)))

    rows = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=groups), start=1):
        for name, model in _classifier_models(random_state + fold).items():
            model.fit(x.iloc[train_idx], y[train_idx])
            prob = model.predict_proba(x.iloc[test_idx])[:, 1]
            pred = model.predict(x.iloc[test_idx])
            rows.append({"fold": fold, "model": name, **_metrics(y[test_idx], pred, prob)})

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("model")
        .agg(
            folds=("fold", "nunique"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            f1_mean=("f1", "mean"),
            accuracy_mean=("accuracy", "mean"),
            brier_mean=("brier", "mean"),
        )
        .reset_index()
        .sort_values("roc_auc_mean", ascending=False)
    )
    frame.to_csv(output_dir / "cross_validation_metrics.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "cross_validation_summary.csv", index=False, encoding="utf-8")
    _plot_cv_summary(summary, output_dir)
    return frame, summary


def _level_difficulty_analysis(
    data: pd.DataFrame,
    output_dir: Path,
    random_state: int,
    test_size: float,
) -> dict[str, Any]:
    features = data.groupby("level_id", as_index=False)[FEATURE_COLUMNS].first()
    targets = (
        data.groupby("level_id", as_index=False)
        .agg(
            actual_difficulty=("difficulty", "mean"),
            mean_success_rate=("success_rate", "mean"),
            agents_tested=("agent", "nunique"),
        )
    )
    level_data = features.merge(targets, on="level_id", how="inner")
    threshold = float(level_data["actual_difficulty"].median())
    level_data["hard_level"] = (level_data["actual_difficulty"] >= threshold).astype(int)

    train_frame, test_frame = train_test_split(
        level_data,
        test_size=test_size,
        random_state=random_state,
        stratify=level_data["hard_level"] if level_data["hard_level"].nunique() > 1 else None,
    )

    x_train = train_frame[FEATURE_COLUMNS]
    x_test = test_frame[FEATURE_COLUMNS]
    y_train = train_frame["actual_difficulty"].to_numpy()
    y_test = test_frame["actual_difficulty"].to_numpy()

    regressors = {
        "level_random_forest": RandomForestRegressor(
            n_estimators=240,
            min_samples_leaf=3,
            random_state=random_state,
        ),
        "level_gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }

    metrics: dict[str, Any] = {}
    predictions = pd.DataFrame(
        {
            "level_id": test_frame["level_id"].to_numpy(),
            "actual_difficulty": y_test,
            "hard_level": test_frame["hard_level"].to_numpy(),
        }
    )

    fitted = {}
    for name, model in regressors.items():
        model.fit(x_train, y_train)
        pred = np.clip(model.predict(x_test), 0.0, 1.0)
        metrics[name] = _regression_metrics(y_test, pred)
        predictions[f"{name}_predicted_difficulty"] = pred
        fitted[name] = model

    best_name = max(regressors, key=lambda key: metrics[key]["r2"])
    best_model = fitted[best_name]
    predictions["predicted_difficulty"] = predictions[f"{best_name}_predicted_difficulty"]

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    metrics["metadata"] = {
        "hard_level_threshold": threshold,
        "levels": int(len(level_data)),
        "best_model": best_name,
    }

    predictions.to_csv(output_dir / "level_difficulty_predictions.csv", index=False, encoding="utf-8")
    importance.to_csv(output_dir / "level_feature_importance.csv", index=False, encoding="utf-8")
    _save_json(output_dir / "level_model_metrics.json", metrics)
    _plot_level_predictions(predictions, output_dir)

    return {
        "metrics": metrics,
        "best_model": best_name,
        "feature_importance": importance,
        "predictions": predictions,
    }


def _importance(best_model: Pipeline, x_test: pd.DataFrame, y_test: np.ndarray, random_state: int) -> pd.DataFrame:
    preprocessor = best_model.named_steps["preprocess"]
    names = _feature_names(preprocessor)
    estimator = best_model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
        importance = pd.DataFrame({"feature": names, "importance": scores})
    elif hasattr(estimator, "coef_"):
        scores = np.abs(estimator.coef_.ravel())
        importance = pd.DataFrame({"feature": names, "importance": scores})
    else:
        perm = permutation_importance(best_model, x_test, y_test, n_repeats=5, random_state=random_state, scoring="roc_auc")
        importance = pd.DataFrame({"feature": MODEL_INPUT_COLUMNS, "importance": perm.importances_mean})

    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def _write_research_summary(
    data: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    best_name: str,
    feature_importance: pd.DataFrame,
    gmm_summary: pd.DataFrame,
    ablation: pd.DataFrame,
    cv_summary: pd.DataFrame,
    level_analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    best = metrics[best_name]
    top_features = ", ".join(feature_importance.head(5)["feature"].astype(str).tolist())
    best_ablation = ablation.iloc[0]
    best_cv = cv_summary.iloc[0]
    hard_cluster = gmm_summary.sort_values("mean_difficulty", ascending=False).iloc[0]
    level_best = level_analysis["best_model"]
    level_metrics = level_analysis["metrics"][level_best]
    level_top_features = ", ".join(level_analysis["feature_importance"].head(5)["feature"].astype(str).tolist())

    text = f"""# Research Summary

## Цель

Оценить сложность процедурно сгенерированных игровых уровней как вероятность непрохождения:

```text
difficulty = 1 - P(success)
```

## Данные

- строк в датасете: {len(data)}
- уникальных уровней: {data["level_id"].nunique()}
- агенты: {", ".join(sorted(data["agent"].unique()))}
- средняя сложность: {data["difficulty"].mean():.3f}

## Проверяемая гипотеза

Структурные признаки уровня, такие как длина кратчайшего пути, доступная площадь, плотность препятствий, тупики и ветвистость маршрутов, позволяют предсказывать вероятность успешного прохождения AI-агентом.

## Основной результат

Лучшая модель: `{best_name}`.

- ROC-AUC: {best["roc_auc"]:.3f}
- F1: {best["f1"]:.3f}
- Accuracy: {best["accuracy"]:.3f}
- Brier score: {best["brier"]:.3f}

## Почему это не просто запуск библиотеки

В проекте самостоятельно задается метрика сложности, генерируется датасет по seed, извлекаются признаки уровней, запускаются собственные агенты, сравниваются несколько ML-подходов и оценивается неопределенность прогноза.

## Ablation study

Лучший набор признаков в ablation study: `{best_ablation["experiment"]}` с ROC-AUC={best_ablation["roc_auc"]:.3f}. Это показывает вклад выбранных групп признаков в проверку гипотезы.

## Group cross-validation

Для проверки устойчивости модели используется GroupKFold по `level_id`, чтобы один и тот же уровень не попадал одновременно в train и test.

- лучшая модель по CV: `{best_cv["model"]}`
- средний ROC-AUC: {best_cv["roc_auc_mean"]:.3f}
- стандартное отклонение ROC-AUC: {best_cv["roc_auc_std"]:.3f}

## Анализ сложности уровня

Отдельно обучена модель, которая предсказывает среднюю сложность уровня только по структурным признакам, без признака `agent`.

- лучшая модель: `{level_best}`
- R2: {level_metrics["r2"]:.3f}
- MAE: {level_metrics["mae"]:.3f}
- RMSE: {level_metrics["rmse"]:.3f}
- главные структурные признаки: {level_top_features}

## EM-кластеризация

Самый сложный кластер: `{int(hard_cluster["em_cluster"])}` со средней difficulty={hard_cluster["mean_difficulty"]:.3f}. EM используется для группировки уровней на типы сложности без заранее заданных классов.

## Объяснимость

Наиболее важные признаки: {top_features}.

Эти признаки можно использовать для объяснения, почему конкретный уровень считается легким или сложным.
"""
    (output_dir / "research_summary.md").write_text(text, encoding="utf-8")


def train_models(
    data: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
    test_size: float = 0.25,
    cv_folds: int = 3,
) -> dict[str, Any]:
    errors = validate_dataset(data, require_target=True)
    if errors:
        raise ValueError("; ".join(errors))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    x = data[MODEL_INPUT_COLUMNS]
    y = data["success"].astype(int).to_numpy()
    groups = data["level_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    metrics: dict[str, dict[str, float]] = {}
    predictions = pd.DataFrame(
        {
            "level_id": data.iloc[test_idx]["level_id"].to_numpy(),
            "agent": data.iloc[test_idx]["agent"].to_numpy(),
            "actual_success": y_test,
        }
    )

    fitted_models: dict[str, Pipeline] = {}
    for name, model in _classifier_models(random_state).items():
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        pred = model.predict(x_test)
        metrics[name] = _metrics(y_test, pred, prob)
        predictions[f"{name}_prob_success"] = prob
        fitted_models[name] = model

    bayes_metrics, uncertainty, bayes_probs = _train_bayesian(x_train, y_train, x_test, y_test, random_state)
    metrics["bayesian_logistic_laplace"] = bayes_metrics
    predictions["bayesian_logistic_laplace_prob_success"] = bayes_probs
    predictions = pd.concat([predictions, uncertainty], axis=1)

    gmm_summary = _gmm_summary(data.iloc[train_idx], output, random_state)
    ablation = _ablation_study(data, train_idx, test_idx, random_state, output)
    _, cv_summary = _cross_validate_models(data, output, random_state, cv_folds)

    best_name = max(metrics, key=lambda key: metrics[key]["roc_auc"])
    best_model = fitted_models.get(best_name)
    if best_model is None:
        best_name = "random_forest"
        best_model = fitted_models[best_name]

    importance = _importance(best_model, x_test, y_test, random_state)
    importance.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8")
    if f"{best_name}_prob_success" in predictions:
        frac_pos, mean_pred = calibration_curve(y_test, predictions[f"{best_name}_prob_success"], n_bins=8, strategy="uniform")
        calibration = pd.DataFrame(
            {
                "mean_predicted_probability": mean_pred,
                "fraction_of_positives": frac_pos,
            }
        )
        calibration.to_csv(output / "calibration_curve.csv", index=False, encoding="utf-8")
        _plot_calibration(calibration, output)

    with (output / "best_model.pkl").open("wb") as fh:
        pickle.dump(best_model, fh)

    _save_json(output / "metrics.json", metrics)
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8")

    _plot_metrics(metrics, output)
    _plot_difficulty_by_density(data, output)
    _plot_uncertainty(predictions, output)
    plot_level_examples(data, output)
    level_analysis = _level_difficulty_analysis(data, output, random_state, test_size)
    _write_research_summary(data, metrics, best_name, importance, gmm_summary, ablation, cv_summary, level_analysis, output)

    return {
        "metrics": metrics,
        "best_model": best_name,
        "feature_importance": importance,
        "predictions": predictions,
        "ablation": ablation,
        "gmm_summary": gmm_summary,
        "cv_summary": cv_summary,
        "level_analysis": level_analysis,
    }
