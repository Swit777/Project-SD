"""Preserved procedural-level prototype; the current research app is app.py."""

from pathlib import Path
import json
import sys
import tempfile

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_ai_complexity.config import load_config
from game_ai_complexity.dataset import (
    INFERENCE_COLUMNS,
    REQUIRED_COLUMNS,
    generate_dataset,
    validate_dataset,
    validate_inference_dataset,
)
from game_ai_complexity.models import train_models
from game_ai_complexity.predict import predict_success


st.set_page_config(page_title="Game AI Complexity", layout="wide")

st.title("Game AI Level Complexity")
st.caption("Оценка сложности уровня как difficulty = 1 - P(success)")

default_data = ROOT / "data" / "processed" / "levels_results.csv"
default_reports = ROOT / "reports"
default_model = default_reports / "best_model.pkl"

uploaded = st.file_uploader("Загрузить CSV", type=["csv"])

if uploaded:
    data = pd.read_csv(uploaded)
    source_name = uploaded.name
elif default_data.exists():
    data = pd.read_csv(default_data)
    source_name = str(default_data)
else:
    cfg = load_config(ROOT / "configs" / "default.json")
    cfg["generation"]["n_seeds"] = 12
    data = generate_dataset(cfg)
    source_name = "demo generated in memory"

has_training_target = all(column in data.columns for column in REQUIRED_COLUMNS)
training_errors = validate_dataset(data, require_target=True) if has_training_target else []
inference_errors = validate_inference_dataset(data)

if inference_errors:
    st.error("CSV не подходит даже для предсказания:")
    for error in inference_errors:
        st.write(f"- {error}")
    st.stop()

st.sidebar.header("Dataset")
st.sidebar.write(source_name)
st.sidebar.metric("Строк", len(data))
st.sidebar.metric("Уровней", data["level_id"].nunique())
st.sidebar.write("Режим: обучение" if has_training_target else "Режим: предсказание")
if has_training_target:
    st.sidebar.metric("Средняя difficulty", f"{data['difficulty'].mean():.3f}")

tab_data, tab_models, tab_predict, tab_level, tab_schema = st.tabs(
    ["Данные", "Модели", "Предсказание", "Сложность уровня", "Схема"]
)

with tab_data:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Первые строки")
        st.dataframe(data.head(100), use_container_width=True)
    with right:
        st.subheader("Проверка")
        if has_training_target and not training_errors:
            st.success("Датасет подходит для обучения и предсказания.")
        elif has_training_target and training_errors:
            st.warning("Датасет подходит для предсказания, но не для обучения.")
            for error in training_errors:
                st.write(f"- {error}")
        else:
            st.info("Датасет подходит для предсказания. Для обучения нужны целевые колонки.")

        st.subheader("Агенты")
        st.dataframe(data["agent"].value_counts().rename_axis("agent").reset_index(name="rows"), use_container_width=True)

    if has_training_target:
        chart_data = (
            data.groupby(["requested_obstacle_density", "agent"], as_index=False)["difficulty"]
            .mean()
            .pivot(index="requested_obstacle_density", columns="agent", values="difficulty")
        )
        st.line_chart(chart_data)

with tab_models:
    if not has_training_target:
        st.info("Для обучения моделей загрузите CSV с колонками success, success_rate, steps, reward, episodes и difficulty.")
    else:
        metrics_path = default_reports / "metrics.json"
        if metrics_path.exists() and not uploaded:
            metrics = pd.DataFrame(json.loads(metrics_path.read_text(encoding="utf-8"))).T
            st.subheader("Готовые метрики")
            st.dataframe(metrics.sort_values("roc_auc", ascending=False), use_container_width=True)

        if st.button("Обучить модели на текущем датасете", type="primary"):
            with tempfile.TemporaryDirectory() as tmp:
                result = train_models(data, Path(tmp), random_state=42, test_size=0.25)
                metrics = pd.DataFrame(result["metrics"]).T.sort_values("roc_auc", ascending=False)
                st.subheader("Метрики текущего запуска")
                st.dataframe(metrics, use_container_width=True)
                st.subheader("Важность признаков")
                st.dataframe(result["feature_importance"].head(20), use_container_width=True)

        plot = default_reports / "plots" / "model_comparison.png"
        if plot.exists() and not uploaded:
            st.image(str(plot), caption="Сравнение моделей")
        cv_plot = default_reports / "plots" / "cross_validation.png"
        if cv_plot.exists() and not uploaded:
            st.image(str(cv_plot), caption="Group cross-validation по level_id")
        calibration_plot = default_reports / "plots" / "calibration_curve.png"
        if calibration_plot.exists() and not uploaded:
            st.image(str(calibration_plot), caption="Calibration curve")

with tab_predict:
    if not default_model.exists():
        st.warning("Сначала обучите модель: python .\\run_pipeline.py run-all")
    else:
        predictions = predict_success(data[INFERENCE_COLUMNS].copy(), default_model)
        st.subheader("Предсказания сохраненной модели")
        st.dataframe(predictions.head(100), use_container_width=True)

        csv = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Скачать predictions.csv",
            csv,
            "new_dataset_predictions.csv",
            "text/csv",
        )

with tab_level:
    level_metrics_path = default_reports / "level_model_metrics.json"
    level_importance_path = default_reports / "level_feature_importance.csv"
    if level_metrics_path.exists() and not uploaded:
        level_metrics = json.loads(level_metrics_path.read_text(encoding="utf-8"))
        st.subheader("Модель средней сложности уровня")
        st.json(level_metrics)
    if level_importance_path.exists() and not uploaded:
        st.subheader("Структурные признаки уровня")
        st.dataframe(pd.read_csv(level_importance_path).head(15), use_container_width=True)
    plot = default_reports / "plots" / "level_predicted_vs_actual.png"
    if plot.exists() and not uploaded:
        st.image(str(plot), caption="Предсказанная и фактическая сложность уровня")
    examples_plot = default_reports / "plots" / "level_examples.png"
    if examples_plot.exists() and not uploaded:
        st.image(str(examples_plot), caption="Примеры easy / medium / hard уровней")

with tab_schema:
    st.subheader("Колонки для предсказания")
    st.code("\n".join(INFERENCE_COLUMNS), language="text")
    st.subheader("Дополнительные колонки для обучения")
    st.code("\n".join(column for column in REQUIRED_COLUMNS if column not in INFERENCE_COLUMNS), language="text")
    st.write(
        "Если профессор загрузит новый CSV с входными колонками, приложение рассчитает вероятность прохождения. "
        "Если в CSV есть еще и целевые колонки, можно заново обучить модели и сравнить метрики."
    )
