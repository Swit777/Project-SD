from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from player_experience.config import load_config
from player_experience.features import validate_dataset
from player_experience.models import predict_dataset, train_experiment
from player_experience.reports import VERDICTS, write_reports


REPORTS = ROOT / "reports/player_experience"
DATA = ROOT / "data/processed/player_experience.csv"
MODEL = REPORTS / "models/best_model.joblib"
st.set_page_config(page_title="Player Experience Lab (Legacy)", page_icon=":material/analytics:", layout="wide")
st.markdown("""<style>
    .block-container {max-width: 1360px; padding-top: 2rem; padding-bottom: 2rem;}
    h1 {font-size: 2rem !important; letter-spacing: 0 !important;}
    h2 {font-size: 1.3rem !important; letter-spacing: 0 !important;}
    h3 {font-size: 1.05rem !important; letter-spacing: 0 !important;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    [data-testid="stMetric"] {border-left: 3px solid #5ed3b6; padding-left: 16px;}
    button, [data-testid="stFileUploader"] {border-radius: 6px !important;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def read_csv(path: str, modified: int) -> pd.DataFrame:
    return pd.read_csv(path)


def csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8-sig")


def chart_style(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#15171c", plot_bgcolor="#15171c",
                      font=dict(family="Arial", color="#eef0f3", size=13),
                      margin=dict(l=10, r=15, t=35, b=10), height=310)
    return fig


def show_results(directory: Path):
    metadata = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(directory / "metrics.csv").sort_values("validation_player_mae")
    primary = metadata["hypothesis"]
    best = metadata["test_metrics"][metadata["best_model"]]
    cols = st.columns(4)
    cols[0].metric("Test MAE по игрокам", f"{best['player_mae']:.2f}")
    cols[1].metric("Преимущество динамики, Δ MAE", f"{primary['delta_mae']:+.3f}")
    cols[2].metric("95% интервал Δ", f"[{primary['ci_low']:+.2f}; {primary['ci_high']:+.2f}]")
    cols[3].metric("Независимых test-игроков", metadata["splits"]["test"]["players"])
    st.info(VERDICTS[primary["verdict"]], icon=":material/science:")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Сравнение на новых игроках")
        labels = {"mean_baseline": "Среднее", "median_baseline": "Медиана", "ridge_aggregate": "Ridge",
                  "random_forest_aggregate": "RF · агрегаты", "random_forest_dynamic": "RF · динамика",
                  "bayesian_ridge_dynamic": "Bayesian Ridge"}
        plotted = metrics.assign(name=metrics["model"].map(labels))
        fig = px.bar(plotted, x="player_mae", y="name", orientation="h",
                     color="feature_set", color_discrete_map={"aggregate": "#899ba5", "dynamic": "#5ed3b6"},
                     labels={"player_mae": "MAE по игрокам", "name": "", "feature_set": "Признаки"})
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
        st.plotly_chart(chart_style(fig), width="stretch", key=f"comparison_{directory}")
    with right:
        st.subheader("Выбор по validation")
        st.write(metadata["best_model"])
        split_table = pd.DataFrame(metadata["splits"]).T.rename(columns={"players": "Игроки", "observations": "Ответы"})
        st.dataframe(split_table, width="stretch")
        st.caption("Игроки между частями не пересекаются. Положительное Δ означает меньшую ошибку с динамикой.")
        st.caption(f"R² выбранной модели: {best['r2']:.3f}. Покрытие дескриптивного интервала: {metadata['interval']['test_coverage']:.1%}.")
    st.dataframe(metrics.rename(columns={"player_mae": "Test MAE по игрокам", "validation_player_mae": "Validation MAE по игрокам"}),
                 hide_index=True, width="stretch")
    st.download_button("Отчёт исследования", (directory / "research_report_ru.md").read_bytes(),
                       file_name="research_report_ru.md", icon=":material/download:", key=f"report_{directory}")
    st.download_button("Результаты CSV", csv_bytes(metrics), file_name="metrics.csv", icon=":material/download:", key=f"metrics_{directory}")


st.caption("PLAYER EXPERIENCE LAB / POWERWASH SIMULATOR")
st.title("Оценка игрового опыта")
st.write("Самооценка удовольствия и динамика действий игрока")

if not DATA.exists() or not (REPORTS / "run_manifest.json").exists():
    st.warning("Данные или результаты эксперимента ещё не подготовлены.")
    st.code("python run_pipeline.py run-all", language="shell")
    st.stop()

data = read_csv(str(DATA), DATA.stat().st_mtime_ns)
config = load_config()
windows = config["features"]["windows_minutes"]
with st.sidebar:
    st.subheader("Исследование")
    st.metric("Наблюдения", f"{len(data):,}")
    st.metric("Игроки", f"{data.player_id.nunique():,}")
    st.divider()
    st.markdown("[Открытый набор данных · CC0](https://doi.org/10.17605/OSF.IO/WPEH6)")
    st.markdown("[Статья создателей набора](https://www.nature.com/articles/s41597-023-02530-3)")
    st.caption("Цель: Enjoyment, 0–100. Одна игра, исследовательская версия. Причинный эффект адаптации не проверяется.")

results_tab, data_tab, features_tab, upload_tab = st.tabs(["Результаты", "Данные", "Признаки", "Новый датасет"])
with results_tab:
    show_results(REPORTS)

with data_tab:
    a, b = st.columns(2)
    modes = a.multiselect("Режим", sorted(data["mode"].unique()), default=sorted(data["mode"].unique()), key="modes_filter")
    jobs = b.multiselect("Уровень", sorted(data["job"].unique()), key="jobs_filter")
    filtered = data[data["mode"].isin(modes)]
    if jobs:
        filtered = filtered[filtered["job"].isin(jobs)]
    st.caption(f"{len(filtered):,} ответов · {filtered.player_id.nunique():,} игроков")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(chart_style(px.histogram(filtered, x="enjoyment", nbins=25,
            color_discrete_sequence=["#5ed3b6"], labels={"enjoyment": "Удовольствие, 0–100"})), width="stretch")
    with right:
        st.subheader("Аудит подготовки")
        audit = json.loads((DATA.parent / "player_experience_audit.json").read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame({"Этап": ["Выбранные ответы", "Без активного эпизода", "Несовпадение контекста", "В анализе"],
            "Ответы": [audit["input_responses"], audit["missing_episode"], audit["episode_context_mismatch"], audit["prepared_responses"]]}),
            hide_index=True, width="stretch")
        st.caption("Исключение ответов может смещать выборку. События из будущего и с той же секундой не используются.")
    st.dataframe(filtered, hide_index=True, width="stretch", height=330)
    st.download_button("Выгрузить выборку", csv_bytes(filtered), file_name="player_experience_filtered.csv", icon=":material/download:")

with features_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Важность признаков · RF dynamic")
        importance = pd.read_csv(REPORTS / "feature_importance.csv").head(12).iloc[::-1]
        fig = px.bar(importance, x="mae_increase", y="feature", orientation="h", color_discrete_sequence=["#f2bc6c"],
                     labels={"mae_increase": "Рост MAE при перестановке", "feature": ""})
        st.plotly_chart(chart_style(fig), width="stretch")
        st.caption("Не причинное объяснение. При коррелирующих признаках важность может распределяться между ними.")
    with right:
        st.subheader("Временные окна · разведочный анализ")
        st.dataframe(pd.read_csv(REPORTS / "window_ablation.csv"), hide_index=True, width="stretch")
        st.caption("Один и тот же Random Forest: общие признаки + одно окно. Это не основной тест гипотезы.")
        st.subheader("GroupKFold · только development")
        st.dataframe(pd.read_csv(REPORTS / "cross_validation.csv"), hide_index=True, width="stretch")

with upload_tab:
    operation = st.segmented_control("Режим анализа", ["Прогноз", "Новый эксперимент"], default="Прогноз", key="upload_mode")
    st.download_button("Пример входного CSV", (REPORTS / "example_prediction_input.csv").read_bytes(),
                       file_name="example_prediction_input.csv", icon=":material/download:")
    upload = st.file_uploader("Подготовленная таблица признаков", type=["csv"])
    if upload is not None:
        try:
            payload = upload.getvalue()
            incoming = pd.read_csv(io.BytesIO(payload))
        except (ValueError, UnicodeError) as error:
            st.error(f"CSV не прочитан: {error}")
        else:
            training = operation == "Новый эксперимент"
            errors = validate_dataset(incoming, windows, require_target=training)
            if errors:
                st.error("Набор не соответствует схеме: " + "; ".join(errors))
            else:
                st.dataframe(incoming.head(10), hide_index=True, width="stretch")
                st.caption(f"{len(incoming):,} строк · происхождение и семантика входных данных не проверены автоматически")
                input_key = hashlib.sha256(payload + str(operation).encode()).hexdigest()
                if st.button("Запустить эксперимент" if training else "Рассчитать прогноз", type="primary", icon=":material/play_arrow:", key="analyze_upload"):
                    try:
                        with st.spinner("Вычисление..."):
                            if training:
                                directory = REPORTS / "uploads" / input_key[:16]
                                directory.mkdir(parents=True, exist_ok=True)
                                input_path = directory / "input.csv"
                                input_path.write_bytes(payload)
                                result = train_experiment(incoming, directory, config, input_path)
                                write_reports(incoming, result, directory)
                                st.session_state["upload_result"] = (input_key, str(directory))
                            else:
                                prediction = predict_dataset(incoming, MODEL)
                                st.session_state["upload_prediction"] = (input_key, prediction)
                    except (ValueError, OSError) as error:
                        st.error(str(error))
                saved = st.session_state.get("upload_prediction")
                if not training and saved and saved[0] == input_key:
                    st.dataframe(saved[1], hide_index=True, width="stretch")
                    st.download_button("Прогнозы CSV", csv_bytes(saved[1]), file_name="predictions.csv", icon=":material/download:")
                    st.caption("Интервалы дескриптивные, без гарантии покрытия. Перенос на другую игру не проверен.")
                experiment = st.session_state.get("upload_result")
                if training and experiment and experiment[0] == input_key:
                    show_results(Path(experiment[1]))
