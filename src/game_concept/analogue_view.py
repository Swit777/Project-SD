from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .analogues import find_analogues, owner_evidence
from .charts import analogue_landscape, mechanic_matrix
from .live_reviews import refresh_review_totals
from .mechanics import KEYS, LABELS
from .sources import sha256_file
from .dataset import GENRE_COLUMNS, MECHANIC_COLUMNS
from .tag_similarity import FORMAT_TAGS, TagIndex


@st.cache_data
def discovery_catalog(path: str, modified: int, manifest_modified: int):
    source = Path(path)
    meta = json.loads(source.with_name("analogue_catalog_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(source) != meta["catalog_sha256"]:
        raise ValueError("Контрольная сумма каталога аналогов не совпадает")
    return pd.read_csv(source)


@st.cache_resource(max_entries=2)
def tag_index(path, modified, manifest_modified):
    return TagIndex(discovery_catalog(path, modified, manifest_modified))


def show_analogues(root: Path, features, bundle, fallback_catalog, chart):
    st.divider()
    st.subheader("Сопоставимые игры")
    source_mode = st.segmented_control("Источник аналогов", ["Рыночный каталог", "Только train"],
                                       default="Рыночный каталог", key="analogue_source")
    profile = st.segmented_control("Подбор", ["Ближайшие", "Рыночные референсы", "Строгий"], default="Ближайшие", key="analogue_profile_mode")
    reviews_default, coverage_default, price_default, modes_default, order_default, suffix = {
        "Ближайшие": (30, 0.5, 0.0, True, "Сходство", "close"),
        "Рыночные референсы": (100, 0.25, 0.0, False, "Больше отзывов", "market"),
        "Строгий": (30, 1.0, 3.0, True, "Сходство", "strict")}[profile]
    path = root / "data/processed/game_concept/analogue_catalog.csv"
    meta_path = path.with_name("analogue_catalog_manifest.json")
    has_index = path.exists() and meta_path.exists()
    market = discovery_catalog(str(path), path.stat().st_mtime_ns, meta_path.stat().st_mtime_ns) if has_index else fallback_catalog
    reference = bundle["reference"] if source_mode == "Только train" else market
    with st.expander("Формат игры и ориентир"):
        styles = st.multiselect("Теги формата", FORMAT_TAGS, key="analogue_tags")
        strict_tags = st.checkbox("Все выбранные теги обязательны", value=False, key="analogue_strict_tags")
        anchor_search = st.text_input("Поиск игры-ориентира", key="analogue_anchor_query").strip()
        choices = market[market.name.str.contains(anchor_search, case=False, regex=False, na=False) | market.appid.astype(str).eq(anchor_search)].head(40) if len(anchor_search) >= 2 else market.iloc[:0]
        anchor = st.selectbox("Игра-ориентир", [None] + choices.appid.tolist(),
            format_func=lambda v: "Без ориентира" if v is None else choices.loc[choices.appid.eq(v), "name"].iloc[0], key="analogue_anchor")
    query_tags = list(styles)
    if anchor is not None:
        query_tags += json.loads(choices.loc[choices.appid.eq(anchor), "tags_json"].iloc[0])
    scores = None
    if query_tags:
        index = tag_index(str(path), path.stat().st_mtime_ns, meta_path.stat().st_mtime_ns) if has_index else TagIndex(market)
        scores = index.scores(query_tags)
        if scores is None:
            st.warning("Для выбранного ориентира нет информативных тегов; используется структурное сходство.")
    fields = st.columns([1, 1.35, 1, 1.1])
    min_reviews = fields[0].selectbox("Минимум отзывов в снимке", [0, 10, 30, 100, 500], index=[0, 10, 30, 100, 500].index(reviews_default), key=f"analogue_reviews_{suffix}")
    coverage = fields[1].slider("Совпадение выбранных механик", 0.25, 1.0, coverage_default, 0.25, format="%.2f", key=f"analogue_coverage_{suffix}")
    price_value = fields[2].selectbox("Допуск цены", [0.0, 2.0, 3.0], index=[0.0, 2.0, 3.0].index(price_default),
        format_func=lambda x: "Любая цена" if x == 0 else f"В пределах ×{x:g}", key=f"analogue_price_{suffix}")
    price_factor = price_value or None
    require_modes = fields[3].checkbox("Обязательные Co-op / PvP", value=modes_default, key=f"analogue_modes_{suffix}")
    fields[3].caption("Подтверждение категориями Steam")
    order = st.selectbox("Порядок", ["Сходство", "Больше отзывов"], index=["Сходство", "Больше отзывов"].index(order_default), key=f"analogue_sort_{suffix}")
    settings = dict(min_reviews=min_reviews, min_coverage=coverage, require_modes=require_modes, price_factor=price_factor,
                    tag_scores=scores, required_tags=tuple(styles) if strict_tags else (), exclude_appid=anchor)
    matches, diagnostics = find_analogues(features, reference, limit=None, **settings)
    if order == "Больше отзывов":
        matches = matches.sort_values(["review_count", "similarity", "appid"], ascending=[False, False, True]).reset_index(drop=True)
    diagnostics.update(profile=profile, order=order, anchor_appid=anchor, query_tags=styles)
    if not has_index and source_mode != "Только train":
        st.warning("Расширенный каталог ещё не подготовлен; используются игры исследовательского корпуса.")
    if profile == "Рыночные референсы":
        st.info("Расширенный поиск: неполное совпадение механик и режимов допустимо. Это заметные рыночные референсы, не прямые аналоги. "
                "Количество отзывов характеризует публичный отклик, а не коммерческий успех.")
    if not query_tags:
        st.caption("Без формата или игры-ориентира сходство ограничено широкими жанрами и извлечёнными механиками.")
    st.session_state["analogue_diagnostics"] = diagnostics
    st.session_state["analogue_results"] = matches.head(24)
    cols = st.columns(4)
    cols[0].metric("Игр в каталоге", f"{diagnostics['reference_games']:,}")
    cols[1].metric("Жанр, механики и цена", f"{diagnostics['matched_before_reviews']:,}")
    cols[2].metric("Прошли фильтр отзывов", f"{diagnostics['eligible_games']:,}")
    cols[3].metric("Отсеяны по отзывам", f"{diagnostics['excluded_low_or_unknown_reviews']:,}")
    st.caption("Рыночные аналоги не меняют прогноз модели. Отбор по отзывам смещает список к более заметным играм. "
               "Сходство — балл признаков, не вероятность успеха. Отсутствие свидетельства механики не доказывает её отсутствие в игре.")
    if matches.empty:
        st.warning("Подходящих аналогов с этими условиями нет. Слабые совпадения в список не добавлены.")
        return
    query = st.text_input("Название или AppID среди аналогов", key="analogue_search").strip()
    visible = matches[matches.name.str.contains(query, case=False, regex=False, na=False) | matches.appid.astype(str).eq(query)] if query else matches
    if visible.empty:
        st.info("В выбранной группе аналогов такой игры нет.")
        return
    shown = visible.head(12).copy()
    st.caption(f"Показаны {len(shown)} из {len(visible):,}; порядок: {order.lower()}. Версия снимка: {shown.iloc[0].get('snapshot_date', 'не указана')}.")
    selected = [k for k in KEYS if features.iloc[0][f"mechanic_{k}"] == 1]
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**Цена и отзывы сопоставимых игр**")
        st.plotly_chart(chart(analogue_landscape(matches.sort_values('similarity', ascending=False), float(np.expm1(features.iloc[0].log_price))), 380),
                        width="stretch", key="analogue_landscape")
        st.caption(f"До 2 000 ближайших из {len(matches):,} совпадений; отзывы снимка, не продажи. Ноль — значение источника, не доказательство отсутствия аудитории.")
    with right:
        st.markdown("**Из чего складывается сходство**")
        profile = shown.head(8)[["name", "mechanic_coverage", "genre_similarity", "price_similarity", "age_similarity"]].copy()
        if scores is not None:
            profile["Теги"] = shown.head(8).tag_similarity.to_numpy()
        profile["name"] = profile.name.str.slice(0, 28)
        profile = profile.rename(columns={"mechanic_coverage": "Механики", "genre_similarity": "Жанры", "price_similarity": "Цена", "age_similarity": "Возраст"})
        profile = profile.melt(id_vars="name", var_name="Признаки", value_name="Сходство")
        fig = px.bar(profile, x="Сходство", y="name", color="Признаки", barmode="group", orientation="h",
                     color_discrete_sequence=["#5ed3b6", "#9db7d5", "#dfb966", "#c989a0", "#dd886b"], labels={"name": ""})
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, 1])
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(chart(fig, 380), width="stretch", key="analogue_profile")
    shown["owner_note"] = [owner_evidence(lo, hi)[1] for lo, hi in zip(shown.owners_lower, shown.owners_upper)]
    display = shown[["name", "similarity", "review_count", "positive_share", "price_usd", "release_date", "owner_note"]].rename(columns={
        "name": "Игра", "similarity": "Сходство, баллы", "review_count": "Отзывы снимка", "positive_share": "Положительные",
        "price_usd": "Цена, USD", "release_date": "Релиз", "owner_note": "Оценка владельцев"})
    st.dataframe(display, hide_index=True, width="stretch", column_config={
        "Сходство, баллы": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        "Цена, USD": st.column_config.NumberColumn(format="%.2f"),
        "Положительные": st.column_config.NumberColumn(format="percent")})
    with st.expander("Матрица совпадений механик"):
        matrix = mechanic_matrix(shown, selected)
        if matrix is not None:
            st.plotly_chart(chart(matrix, 500), width="stretch", key="analogue_matrix")
            st.caption("1 — найдено свидетельство; 0 — свидетельство не найдено. Сначала механики концепции, затем дополнительные элементы аналогов.")
        else:
            st.info("Нет извлечённых механик для матрицы.")
    selected_id = st.selectbox("Подробности игры", shown.appid.tolist(),
        format_func=lambda value: shown.loc[shown.appid.eq(value), "name"].iloc[0], key="analogue_detail")
    item = shown.loc[shown.appid.eq(selected_id)].iloc[0]
    media, details = st.columns([1, 2.4], gap="large")
    if isinstance(item.get("header_image"), str) and item.header_image.startswith("https://"):
        media.image(item.header_image, width="stretch")
    media.link_button("Steam", f"https://store.steampowered.com/app/{int(selected_id)}/", icon=":material/open_in_new:")
    details.markdown(f"**{item['name']}**")
    details.write("Жанры: " + ", ".join(json.loads(item.genres_json)))
    if isinstance(item.get("tags_json"), str):
        details.caption("Теги Steam: " + ", ".join(json.loads(item.tags_json)[:15]))
    details.write("Совпало: " + (", ".join(LABELS[k] for k in json.loads(item.matched_mechanics)) or "Механики не заданы"))
    if any(key in selected for key in ["co_op", "pvp"]) and require_modes:
        details.caption("Выбранные сетевые режимы подтверждены категориями Steam.")
    missing = json.loads(item.missing_mechanics)
    if missing:
        details.warning("Не подтверждено: " + ", ".join(LABELS[k] for k in missing))
    if not require_modes and any(k in selected for k in ["co_op", "pvp"]):
        from .analogues import structured_mode
        unconfirmed = [k for k in ["co_op", "pvp"] if k in selected and not structured_mode(item.get("categories_json"), k)]
        if unconfirmed:
            details.warning("Отличается режим игры: категории Steam не подтверждают " + ", ".join(LABELS[k] for k in unconfirmed))
    details.caption(f"Снимок: {item.get('snapshot_date', 'не указан')}; возраст: {item.age_days / 365.25:.1f} лет; цена без скидки: ${item.price_usd:.2f}.")
    details.caption(item.owner_note)
    if owner_evidence(item.owners_lower, item.owners_upper)[0] == "low_resolution":
        details.warning("Нижний диапазон SteamSpy имеет низкую точность. Его нельзя читать как 20 000 покупателей.")
    path = root / f"data/processed/game_concept/review_totals/{int(selected_id)}.json"
    if details.button("Обновить отзывы Steam", icon=":material/refresh:", key="analogue_refresh"):
        try:
            with st.spinner("Проверка публичной сводки Steam…"):
                refresh_review_totals(root, int(selected_id))
        except (ValueError, OSError, RuntimeError) as error:
            details.error(f"Сводка не обновлена: {error}")
    if path.exists():
        live = json.loads(path.read_text(encoding="utf-8"))
        details.success(f"Проверено Steam: {live['total_reviews']:,} отзывов; положительных {live['total_positive']:,}. "
                        f"Получено {live['captured_at_utc'][:19]} UTC.")
        details.caption("Все языки и типы приобретения, без off-topic. Другая дата и правила подсчёта; сводка не заменяет метки снимка и не переобучает модель.")
    export = visible.drop(columns=GENRE_COLUMNS + MECHANIC_COLUMNS, errors="ignore").copy()
    export["owner_note"] = [owner_evidence(lo, hi)[1] for lo, hi in zip(export.owners_lower, export.owners_upper)]
    st.download_button("Аналоги CSV", export.to_csv(index=False).encode("utf-8-sig"), file_name="concept_analogues.csv", icon=":material/download:")
