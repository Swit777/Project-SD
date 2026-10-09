from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from game_concept.config import load_config
from game_concept.annotation import create_annotation_tasks, current_annotation, evaluate_annotations, load_annotation, save_reviews
from game_concept.dataset import BAND_LABELS, GAME_GENRES, feature_columns, validate_dataset
from game_concept.mechanics import KEYS, LABELS, plain_text
from game_concept.models import train_experiment
from game_concept.predict import compare_components, concept_features, explain_lime, forecast, load_bundle
from game_concept.reports import VERDICTS, write_reports
from game_concept.scraper import scrape_games
from game_concept.analogue_view import show_analogues
from game_concept.analogues import RETRIEVAL_VERSION, review_stats
from game_concept.charts import classification_errors, component_deltas

DATA = ROOT / "data/processed/game_concept"
REPORTS = ROOT / "reports/game_concept"
MODEL = REPORTS / "models/concept_model.joblib"
st.set_page_config(page_title="Game Concept Lab", page_icon=":material/analytics:", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1380px; padding-top: 1.8rem; padding-bottom: 2rem;}
h1 {font-size: 2rem !important; letter-spacing: 0 !important;}
h2 {font-size: 1.3rem !important; letter-spacing: 0 !important;}
h3 {font-size: 1.05rem !important; letter-spacing: 0 !important;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
button, [data-testid="stFileUploader"] {border-radius: 6px !important;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def cached_model(path: str, modified: int):
    return load_bundle(Path(path))


@st.cache_data
def read_csv(path: str, modified: int):
    return pd.read_csv(path)


def table(path: Path):
    return read_csv(str(path), path.stat().st_mtime_ns)


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def chart(fig, height=300):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#15171c", plot_bgcolor="#15171c",
                      font=dict(family="Arial", size=13), height=height,
                      margin=dict(l=5, r=10, t=25, b=5))
    return fig


def show_experiment(directory: Path):
    meta = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = table(directory / "market_metrics.csv")
    hypothesis = meta["hypothesis"]
    st.info(VERDICTS[hypothesis["verdict"]], icon=":material/science:")
    cols = st.columns(3)
    cols[0].metric("Разность ошибок: additive − interactions", f"{hypothesis['delta_log_loss']:+.4f}")
    cols[1].metric("95% bootstrap-интервал", f"[{hypothesis['ci_low']:+.4f}; {hypothesis['ci_high']:+.4f}]")
    cols[2].metric("Группы разработчиков в test", hypothesis["groups"])
    st.caption(f"Выбор модели по validation: {meta['selected_model']}. Test не участвует в выборе.")
    st.plotly_chart(chart(px.bar(metrics, x="model", y="group_log_loss", color="model",
        color_discrete_map={"prior": "#74828b", "genre_only": "#86acc7", "logistic_additive": "#ceab60", "hgb_additive": "#5ed3b6", "hgb_interactions": "#bf728d"},
        labels={"group_log_loss": "Test log loss по разработчикам", "model": "Модель"})), width="stretch")
    st.dataframe(metrics, hide_index=True, width="stretch")
    st.subheader("Какие игры модель распознаёт хуже")
    errors, matrix = classification_errors(table(directory / "test_predictions.csv"), meta["selected_model"])
    left, right = st.columns([1.4, 1])
    left.plotly_chart(chart(errors, 370), width="stretch", key=f"errors_{directory.name}")
    row_totals = matrix.sum(axis=1)
    recall = np.divide(matrix.diagonal(), row_totals, out=np.zeros(4), where=row_totals != 0)
    right.dataframe(pd.DataFrame({"Диапазон": BAND_LABELS, "Игр в test": row_totals, "Полнота, %": np.round(recall * 100, 1)}),
                    hide_index=True, width="stretch")
    selected_metrics = metrics.loc[metrics.model.eq(meta["selected_model"])].iloc[0]
    right.metric("Balanced accuracy", f"{selected_metrics.balanced_accuracy:.1%}")
    right.caption("Обычная accuracy скрывает перекос классов. Слабое распознавание редких крупных игр ограничивает практическую надёжность прогноза.")
    reliability_path = directory / "reliability_bins.csv"
    if reliability_path.exists():
        with st.expander("Проверка вероятностей · калибровка"):
            threshold = st.selectbox("Граница аудитории", [20000, 100000, 500000],
                                    format_func=lambda v: f"≥ {v:,} владельцев", key=f"calibration_{directory.name}")
            curves = table(reliability_path)
            curves = curves[curves.model.eq(meta["selected_model"]) & curves.threshold.eq(threshold) & curves.games.gt(0)].copy()
            curves["Этап"] = curves.stage.map({"before": "До калибровки", "after": "После калибровки"})
            fig = px.line(curves, x="mean_probability", y="observed_fraction", color="Этап", markers=True,
                          color_discrete_map={"До калибровки": "#74828b", "После калибровки": "#5ed3b6"},
                          hover_data=["games", "developer_groups"], labels={"mean_probability": "Средний прогноз", "observed_fraction": "Наблюдаемая доля"})
            fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(color="#74828b", dash="dot"))
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(chart(fig, 360), width="stretch")
            diagnostics = table(directory / "calibration_metrics.csv")
            st.dataframe(diagnostics[diagnostics.model.eq(meta["selected_model"]) & diagnostics.threshold.eq(threshold)], hide_index=True, width="stretch")
            st.caption("Независимый test, 10 фиксированных интервалов; ECE и Brier усреднены по играм. "
                       "Калибровка не гарантирует улучшения. Малые группы точек ненадёжны; это не доверительный интервал нового продукта.")
    st.subheader("Игровое время")
    playtime = meta["playtime"]
    if playtime["status"] == "available":
        time_metrics = table(directory / "playtime_metrics.csv")
        st.dataframe(time_metrics, hide_index=True, width="stretch")
        st.caption(f"Покрытие номинального 90% интервала на test: {playtime['empirical_test_coverage']:.1%}. "
                   "Только игры с доступной положительной оценкой часов; это не D7/D30-retention.")
    else:
        st.warning("Данных игровых часов недостаточно.")
    with st.expander("Временная проверка и воспроизводимость"):
        st.dataframe(table(directory / "temporal_backtest.csv"), hide_index=True, width="stretch")
        st.warning("Проверка ретроспективная: описания и цены могли измениться после релиза. "
                   "Настоящий pre-launch-прогноз и будущие годовые продажи этим тестом не подтверждены.")
        st.json(meta)
    with st.expander("Важность признаков · permutation"):
        importance = table(directory / "feature_importance.csv").sort_values("log_loss_increase", ascending=False).head(15).copy()
        importance["Признак"] = importance.feature.map(lambda c: LABELS.get(c.removeprefix("mechanic_"), c))
        st.plotly_chart(chart(px.bar(importance, x="log_loss_increase", y="Признак", error_x="std", orientation="h",
            labels={"log_loss_increase": "Рост log loss после перестановки"}, color_discrete_sequence=["#5ed3b6"]), 450), width="stretch")
        st.dataframe(importance, hide_index=True, width="stretch")
        st.caption("Число механик пересчитывается после перестановки флагов и не оценивается как независимый вход. "
                   "Перестановки могут нарушать наблюдаемое сочетание компонентов; результат не является причинным эффектом.")
    st.download_button("Отчёт исследования", (directory / "research_report_ru.md").read_bytes(),
                       file_name="research_report_ru.md", icon=":material/download:")


st.title("Прогноз игровой концепции")
st.caption("Steam · сочетания игровых элементов · аудитория и вовлечённость")
if not MODEL.exists() or not (DATA / "games.csv").exists():
    st.error("Модель или данные не подготовлены. Команда запуска исследования указана в README.md.")
    st.stop()

bundle = cached_model(str(MODEL), MODEL.stat().st_mtime_ns)
catalog = table(DATA / "games.csv")
tabs = st.tabs(["Концепция", "Данные Steam", "Исследование", "Новый датасет"])

with tabs[0]:
    active_features = None
    controls, results = st.columns([0.95, 1.7], gap="large")
    with controls:
        st.subheader("Параметры продукта")
        genres = st.multiselect("Жанры", GAME_GENRES, default=["Indie", "RPG"], key="concept_genres")
        mechanics = st.multiselect("Игровые элементы", KEYS, default=["crafting", "co_op"],
                                  format_func=LABELS.get, key="concept_mechanics")
        price = st.number_input("Недисконтная цена, USD", min_value=0.49, max_value=500.0,
                                value=14.99, step=1.0, key="concept_price")
        age = st.number_input("Возраст сопоставимых игр, лет", min_value=0.5, max_value=20.0,
                              value=3.0, step=0.5, key="concept_age")
        languages = st.number_input("Число языков", min_value=1, max_value=100, value=2, step=1)
        platforms = st.columns(3)
        windows = platforms[0].checkbox("Windows", value=True)
        mac = platforms[1].checkbox("Mac")
        linux = platforms[2].checkbox("Linux")
        with st.expander("Модель эксперимента"):
            names = list(bundle["all_market"])
            chosen_model = st.selectbox("Модель", names, index=names.index(bundle["metadata"]["selected_model"]), key="concept_model")
            st.caption(f"Выбрана по validation: {bundle['metadata']['selected_model']}")
        run = st.button("Рассчитать концепцию", icon=":material/analytics:", type="primary", key="forecast")

    with results:
        try:
            features = concept_features(genres, mechanics, price, age, int(languages), windows, mac, linux)
            fingerprint = hashlib.sha256(csv_bytes(features) + chosen_model.encode() + RETRIEVAL_VERSION.encode() + str(MODEL.stat().st_mtime_ns).encode()).hexdigest()
            if run:
                st.session_state["concept_result"] = (fingerprint, forecast(features, bundle, chosen_model))
            stored = st.session_state.get("concept_result")
            if stored is None:
                st.session_state["concept_result"] = (fingerprint, forecast(features, bundle, chosen_model))
                stored = st.session_state["concept_result"]
            current = stored[0] == fingerprint
            if not current:
                st.info("Параметры изменены; расчёт ожидает обновления.")
            else:
                active_features = features
                result = stored[1]
                support = result["support"]
                if support["status"] == "abstain":
                    st.warning("Недостаточно похожих примеров: численный прогноз не выдаётся.")
                    st.json(support)
                else:
                    if support["status"] == "limited":
                        st.warning("Некоторые элементы или их пары редко встречаются в обучающей выборке.")
                    st.subheader("Оценочный масштаб аудитории")
                    cols = st.columns(3)
                    cols[0].metric("P(владельцев ≥ 20 тыс.)", f"{result['p_at_least_20k']:.1%}")
                    cols[1].metric("P(владельцев ≥ 100 тыс.)", f"{result['p_at_least_100k']:.1%}")
                    cols[2].metric("P(владельцев ≥ 500 тыс.)", f"{result['p_at_least_500k']:.1%}")
                    prior = bundle["reference"].owners_band.value_counts(normalize=True).reindex(range(4), fill_value=0)
                    bands = pd.DataFrame({"Диапазон": BAND_LABELS, "Концепция": result["probabilities"], "Частота в train": prior.to_numpy()})
                    bands = bands.melt(id_vars="Диапазон", var_name="Оценка", value_name="Вероятность")
                    fig = px.bar(bands, x="Диапазон", y="Вероятность", color="Оценка", barmode="group",
                                 color_discrete_map={"Концепция": "#5ed3b6", "Частота в train": "#74828b"})
                    fig.update_yaxes(tickformat=".0%", range=[0, 1])
                    st.plotly_chart(chart(fig, 260), width="stretch")
                    st.caption("Оценки владельцев SteamSpy, не фактические продажи и не вероятность прибыли. "
                               "Возраст аналогов не задаёт подтверждённый горизонт будущих продаж.")
                    st.caption("0–20k — нижняя категория источника с низкой точностью, а не обещание 20 тыс. покупателей. "
                               "Серые столбцы — исходное распределение обучающей выборки, не прогноз концепции.")
                    if result["playtime"]:
                        hours = result["playtime"]
                        cols = st.columns(2)
                        cols[0].metric("Оценка медианных часов игры", f"{hours['median_hours_estimate']:.1f} ч")
                        cols[1].metric("Калиброванный 90% интервал", f"{hours['low_hours']:.1f}–{hours['high_hours']:.1f} ч")
                        st.caption("Условная оценка для игр с доступными часами; не длительность удержания и не оценка качества.")
                    with st.expander("Разница при изменении компонентов"):
                        if st.button("Сравнить компоненты", icon=":material/compare_arrows:", key="compare"):
                            st.session_state["comparison"] = (fingerprint, compare_components(features, bundle, chosen_model))
                        comparison = st.session_state.get("comparison")
                        if comparison and comparison[0] == fingerprint:
                            st.plotly_chart(chart(component_deltas(comparison[1]), 480), width="stretch", key="component_deltas")
                            shown = comparison[1].copy()
                            shown["mechanic"] = shown.mechanic.map(LABELS)
                            shown["change"] = shown.change.map({"add": "Добавить", "remove": "Убрать"})
                            shown["agreement"] = shown.agreement.map({"mixed": "Разные направления", "all_positive": "Все положительные", "all_negative": "Все отрицательные",
                                "near_zero": "Около нуля", "partly_neutral": "Часть около нуля", "unavailable": "Нет прогноза"})
                            shown = shown.rename(columns={"mechanic": "Компонент", "change": "Изменение", "support_status": "Поддержка",
                                "delta_pp": "Δ выбранной модели, п.п.", "model_min_delta_pp": "Мин. Δ моделей, п.п.", "model_max_delta_pp": "Макс. Δ моделей, п.п.",
                                "agreement": "Согласованность", "full_models": "Моделей", "support_reasons": "Причины отказа"})
                            st.dataframe(shown, hide_index=True, width="stretch")
                            st.download_button("Сравнение CSV", csv_bytes(shown), file_name="component_comparison.csv", icon=":material/download:")
                        st.caption("Δ P(≥20k) при фиксированном контексте. Разброс Logistic / HGB additive / HGB interactions — "
                                   "межмодельное расхождение, не доверительный интервал и не причинный эффект. Значения в пределах ±0.25 п.п. отмечены как около нуля.")
                    with st.expander("Локальное объяснение · LIME"):
                        if st.button("Вычислить LIME", icon=":material/manage_search:", key="lime"):
                            with st.spinner("Расчёт локальной аппроксимации…"):
                                explanation, details = explain_lime(features, bundle, chosen_model)
                            st.session_state["lime_explanation"] = (fingerprint, explanation, details)
                        lime = st.session_state.get("lime_explanation")
                        if lime and lime[0] == fingerprint:
                            st.plotly_chart(chart(px.bar(lime[1], x="local_weight", y="feature_condition", orientation="h",
                                labels={"local_weight": "Локальный вес", "feature_condition": "Условие"},
                                color_discrete_sequence=["#9db7d5"]), 420), width="stretch", key="lime_chart")
                            st.dataframe(lime[1], hide_index=True, width="stretch")
                            st.caption(f"Диапазон 20k–100k; local R² = {lime[2]['local_surrogate_r2']:.3f}. "
                                       "Локальная аппроксимация, не причинный вклад; возмущения могут быть вне наблюдаемых комбинаций.")
        except ValueError as error:
            st.error(str(error))
    if active_features is not None:
        try:
            show_analogues(ROOT, active_features, bundle, catalog, chart)
        except (ValueError, OSError) as error:
            st.error(f"Каталог аналогов недоступен: {error}")

with tabs[1]:
    st.subheader("Корпус игровых элементов")
    evidence = table(DATA / "mechanic_evidence.csv")
    cols = st.columns(3)
    cols[0].metric("Игр в исследовании", f"{len(catalog):,}")
    cols[1].metric("Свидетельств механик", f"{len(evidence):,}")
    live_path = DATA / "live_games.csv"
    live = table(live_path) if live_path.exists() else pd.DataFrame()
    cols[2].metric("Прямой сбор Steam", len(live))
    query = st.text_input("Поиск игры", key="catalog_search")
    genre_filter = st.multiselect("Жанры корпуса", GAME_GENRES, default=GAME_GENRES, key="catalog_genres")
    filtered = catalog[catalog.name.str.contains(query, case=False, regex=False, na=False)]
    filtered = filtered[filtered.genres_json.map(lambda v: bool(set(json.loads(v)) & set(genre_filter)))]
    with st.expander("Распределение и полнота данных", expanded=True):
        if len(filtered):
            left, right = st.columns(2)
            counts = filtered.owners_band.value_counts().reindex(range(4), fill_value=0)
            distribution = pd.DataFrame({"Диапазон SteamSpy": BAND_LABELS, "Игр": counts.to_numpy()})
            left.plotly_chart(chart(px.bar(distribution, x="Диапазон SteamSpy", y="Игр", color_discrete_sequence=["#9db7d5"])),
                             width="stretch", key="dataset_classes")
            reviews = review_stats(filtered)
            reviews["Отзывы"] = pd.cut(reviews.review_count, [-1, 0, 9, 29, 99, 499, float("inf")],
                                      labels=["0", "1–9", "10–29", "30–99", "100–499", "500+"])
            counts = reviews["Отзывы"].value_counts(sort=False).rename_axis("Отзывы").reset_index(name="Игр")
            right.plotly_chart(chart(px.bar(counts, x="Отзывы", y="Игр", color_discrete_sequence=["#dfb966"])),
                              width="stretch", key="dataset_reviews")
            st.caption(f"После фильтров: {len(filtered):,} игр. Неизвестные отзывы: {reviews.review_count.isna().sum():,}; "
                       f"доступное игровое время: {filtered.playtime_median_hours.notna().sum():,}. Данные снимка, не текущая статистика Steam.")
    st.dataframe(filtered[["appid", "name", "price_usd", "genres_json", "mechanics_json", "owners_lower", "owners_upper", "playtime_median_hours"]].head(250),
                 hide_index=True, width="stretch")
    if len(filtered):
        game_id = st.selectbox("Игра для аудита", filtered.appid.head(250).tolist(),
                               format_func=lambda v: f"{v} · {catalog.loc[catalog.appid.eq(v), 'name'].iloc[0]}", key="audit_game")
        st.dataframe(evidence[evidence.appid.eq(game_id)], hide_index=True, width="stretch")
        claim_path = DATA / "feature_claims.csv"
        if claim_path.exists():
            claims = table(claim_path)
            st.caption("Кандидаты особенных возможностей из описания: требуют ручной разметки, не входы модели.")
            st.dataframe(claims.loc[claims.appid.eq(game_id), ["claim", "status"]], hide_index=True, width="stretch")
    else:
        st.info("Нет игр, соответствующих фильтрам.")
    with st.expander("Ручная проверка извлечения"):
        if st.button("Подготовить аудит 40 игр", icon=":material/fact_check:", key="create_annotation"):
            try:
                create_annotation_tasks(ROOT, games=40, seed=load_config()["seed"])
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(str(error))
        annotation_dir = current_annotation(ROOT)
        if annotation_dir:
            try:
                tasks, audit_games, audit_manifest = load_annotation(annotation_dir)
                if audit_manifest["dataset_sha256"] != bundle["metadata"]["input_sha256"]:
                    st.warning("Аудит относится к другой версии корпуса; его метрики не описывают текущую модель.")
                choices = {g["appid"]: g for g in audit_games}
                audit_id = st.selectbox("Проверяемая игра", list(choices), format_func=lambda v: f"{v} · {choices[v]['name']}", key="annotation_game")
                game = choices[audit_id]
                st.link_button("Источник Steam", game["source_url"], icon=":material/open_in_new:")
                source_col, label_col = st.columns([1.05, 1.5], gap="large")
                with source_col:
                    with st.container(height=460, border=False):
                        st.write(plain_text(game["description"]))
                        st.caption("Теги: " + ", ".join(game["tags"]))
                        st.caption("Категории: " + ", ".join(game["categories"]))
                subset = tasks[tasks.appid.eq(audit_id)].copy()
                label_names = {"": "Не рассмотрено", "1": "Подтверждено", "0": "Не подтверждено", "unclear": "Неясно"}
                editor = subset[["task_id", "mechanic", "expert_label", "expert_notes"]].copy()
                editor["mechanic"] = editor.mechanic.map(LABELS)
                editor["expert_label"] = editor.expert_label.map(label_names)
                edited = label_col.data_editor(editor, hide_index=True, width="stretch", height=460,
                    column_order=["mechanic", "expert_label", "expert_notes"], disabled=["mechanic", "task_id"],
                    column_config={"mechanic": "Элемент", "expert_label": st.column_config.SelectboxColumn("Подтверждение в метаданных", options=list(label_names.values()), required=True),
                                   "expert_notes": "Заметки"}, key=f"labels_{annotation_dir.name}_{audit_id}")
                if st.button("Сохранить разметку", icon=":material/save:", key="save_annotation"):
                    edited["expert_label"] = edited.expert_label.map({v: k for k, v in label_names.items()})
                    save_reviews(annotation_dir, edited[["task_id", "expert_label", "expert_notes"]])
                    st.rerun()
                summary, details = evaluate_annotations(tasks, tasks[["task_id", "expert_label", "expert_notes"]])
                st.caption(f"Рассмотрено {summary['reviewed']} из {summary['tasks']} меток; полностью размечено игр: {summary['fully_labeled_games']}. "
                           "Проверяется наличие свидетельств в Steam, не реальное качество механик. Нерассмотренные метки не считаются отрицательными.")
                if summary["reviewed"]:
                    st.json(summary)
                    st.dataframe(details[details.reviewed.gt(0)], hide_index=True, width="stretch")
                st.download_button("Разметка CSV", csv_bytes(tasks[["task_id", "appid", "mechanic", "expert_label", "expert_notes"]]),
                                   file_name="mechanic_reviews.csv", icon=":material/download:")
                imported = st.file_uploader("CSV разметки", type=["csv"], key="annotation_upload")
                if imported is not None and st.button("Импортировать разметку", icon=":material/upload:", key="import_annotation"):
                    save_reviews(annotation_dir, pd.read_csv(io.BytesIO(imported.getvalue()), keep_default_na=False))
                    st.rerun()
            except (ValueError, KeyError, OSError) as error:
                st.error(str(error))
    with st.expander("Прямое извлечение со Steam"):
        appid_input = st.text_input("Steam AppID или адрес страницы", value="413150", key="steam_appid")
        refresh = st.checkbox("Обновить ответы Steam", value=False)
        if st.button("Получить данные игры", icon=":material/cloud_download:", key="scrape"):
            try:
                match = re.fullmatch(r"(?:https://store\.steampowered\.com/app/)?(\d+)(?:/[^?]*)?(?:\?.*)?", appid_input.strip())
                if not match:
                    raise ValueError("Нужен AppID или публичный адрес store.steampowered.com/app/…")
                with st.spinner("Извлечение публичных данных Steam…"):
                    manifest = scrape_games(ROOT, load_config(), [int(match.group(1))], refresh=refresh)
                st.session_state["last_scrape"] = manifest
                st.rerun()
            except (ValueError, OSError, RuntimeError) as error:
                st.error(str(error))
        if "last_scrape" in st.session_state:
            st.json(st.session_state["last_scrape"])
        if len(live):
            missing_html = int(live.page_unavailable_or_gated.sum())
            if missing_html:
                st.warning(f"У {missing_html} игр HTML-страница недоступна или закрыта. Сохранены публичные API-описания; теги страницы неизвестны.")
            st.dataframe(live[["appid", "name", "captured_at_utc", "genres", "mechanics", "feature_claims"]], hide_index=True, width="stretch")
            st.download_button("Скачать прямой сбор", csv_bytes(live), file_name="steam_live_games.csv", icon=":material/download:")
        st.caption("Текущий сбор проверяет описания; он не смешивается со старыми целевыми метками и не меняет обученную модель.")

with tabs[2]:
    st.subheader("Проверка гипотезы о сочетаниях механик")
    show_experiment(REPORTS)

with tabs[3]:
    st.subheader("Совместимые данные")
    mode = st.segmented_control("Назначение", ["Прогноз", "Новый эксперимент"], default="Прогноз", key="upload_mode")
    template = catalog[feature_columns()].head(50) if mode == "Прогноз" else catalog.head(240)
    st.download_button("CSV-шаблон", csv_bytes(template), file_name="concept_input.csv", icon=":material/download:")
    upload = st.file_uploader("CSV", type=["csv"], key="concept_upload")
    if upload is not None:
        try:
            payload = upload.getvalue()
            if len(payload) > 40 * 1024 * 1024:
                raise ValueError("CSV превышает лимит 40 MB")
            uploaded = pd.read_csv(io.BytesIO(payload))
            errors = validate_dataset(uploaded, training=mode == "Новый эксперимент")
            if errors:
                raise ValueError("; ".join(errors))
            upload_key = hashlib.sha256(payload + mode.encode()).hexdigest()
            st.dataframe(uploaded.head(20), hide_index=True, width="stretch")
            if mode == "Прогноз":
                if len(uploaded) > 200:
                    raise ValueError("В одном интерактивном прогнозе допускается до 200 концепций")
                if st.button("Рассчитать CSV", icon=":material/analytics:", key="predict_upload"):
                    rows = []
                    for index in range(len(uploaded)):
                        prediction = forecast(uploaded.iloc[[index]], bundle)
                        row = {"input_row": index + 1, "support_status": prediction["support"]["status"]}
                        if prediction["probabilities"] is not None:
                            row.update({f"p_band_{i}": p for i, p in enumerate(prediction["probabilities"])})
                        if prediction["playtime"]:
                            row.update(prediction["playtime"])
                        rows.append(row)
                    st.session_state["uploaded_predictions"] = (upload_key, pd.DataFrame(rows))
                saved = st.session_state.get("uploaded_predictions")
                if saved and saved[0] == upload_key:
                    st.dataframe(saved[1], hide_index=True, width="stretch")
                    st.download_button("Результаты CSV", csv_bytes(saved[1]), file_name="concept_predictions.csv", icon=":material/download:")
            else:
                st.warning("Источники и значение меток пользовательского CSV не подтверждены. Основной эксперимент не перезаписывается.")
                if st.button("Обучить на новом наборе", icon=":material/model_training:", key="train_upload"):
                    directory = REPORTS / "uploads" / upload_key
                    directory.mkdir(parents=True, exist_ok=True)
                    source = directory / "input.csv"
                    source.write_bytes(payload)
                    with st.spinner("Обучение и независимая проверка…"):
                        experiment = train_experiment(uploaded, load_config(), directory, source)
                        write_reports(uploaded, experiment, directory)
                    st.session_state["uploaded_experiment"] = (upload_key, str(directory))
                saved = st.session_state.get("uploaded_experiment")
                if saved and saved[0] == upload_key:
                    show_experiment(Path(saved[1]))
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as error:
            st.error(str(error))
