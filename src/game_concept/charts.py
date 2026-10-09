from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from .dataset import BAND_LABELS
from .mechanics import KEYS, LABELS


def analogue_landscape(games: pd.DataFrame, price: float):
    shown = games.head(2000).copy()
    shown["review_axis"] = np.log10(shown.review_count + 1)
    fig = px.scatter(shown, x="price_usd", y="review_axis", color="similarity", hover_name="name",
        hover_data={"review_count": True, "positive_share": ":.1%", "similarity": ":.1f", "review_axis": False},
        color_continuous_scale=["#74828b", "#64d8b6"], range_color=[0, 100],
        labels={"price_usd": "Цена без скидки, USD", "review_axis": "Отзывы в снимке · log₁₀(1+n)", "similarity": "Сходство"})
    fig.add_vline(x=price, line_dash="dot", line_color="#eac66d", annotation_text="Цена концепции")
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    return fig


def mechanic_matrix(games: pd.DataFrame, selected: list[str]):
    top = games.head(12)
    extras = [k for k in KEYS if k not in selected and top[f"mechanic_{k}"].sum() > 0]
    extras.sort(key=lambda k: (-top[f"mechanic_{k}"].sum(), k))
    keys = selected + extras[:max(0, 10 - len(selected))]
    if not keys:
        return None
    labels = [f"{str(row['name'])[:35]} · {row['appid']}" for row in top.to_dict("records")]
    matrix = top[[f"mechanic_{k}" for k in keys]].to_numpy()
    fig = go.Figure(go.Heatmap(z=matrix, x=[LABELS[k] for k in keys], y=labels,
        zmin=0, zmax=1, colorscale=[[0, "#292d36"], [1, "#5ed3b6"]], showscale=False,
        hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(tickangle=-30, automargin=True)
    return fig


def classification_errors(predictions: pd.DataFrame, model: str):
    probabilities = predictions[[f"{model}_p{k}" for k in range(4)]].to_numpy()
    matrix = confusion_matrix(predictions.owners_band, probabilities.argmax(axis=1), labels=range(4))
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
    fig = go.Figure(go.Heatmap(z=normalized, x=BAND_LABELS, y=BAND_LABELS, customdata=matrix,
        text=[[f"{n:,}<br>{p:.1%}" for n, p in zip(ns, ps)] for ns, ps in zip(matrix, normalized)],
        texttemplate="%{text}", zmin=0, zmax=1, colorscale=[[0, "#222630"], [1, "#48b8a1"]],
        hovertemplate="Истина: %{y}<br>Прогноз: %{x}<br>Игр: %{customdata}<br>Доля строки: %{z:.1%}<extra></extra>"))
    fig.update_xaxes(title="Предсказанный диапазон")
    fig.update_yaxes(title="Диапазон источника", autorange="reversed")
    return fig, matrix


def component_deltas(comparison: pd.DataFrame):
    shown = comparison.dropna(subset=["delta_pp"]).copy()
    shown["label"] = shown.apply(lambda r: ("+ " if r.change == "add" else "− ") + LABELS.get(r.mechanic, r.mechanic), axis=1)
    shown = shown.reindex(shown.delta_pp.abs().sort_values(ascending=False).head(16).index).sort_values("delta_pp")
    fig = go.Figure()
    for row in shown.itertuples():
        fig.add_trace(go.Scatter(x=[row.model_min_delta_pp, row.model_max_delta_pp], y=[row.label, row.label], mode="lines",
                                line=dict(color="#80929a", width=4), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=shown.delta_pp, y=shown.label, mode="markers", showlegend=False,
        marker=dict(size=9, color="#5ed3b6"), hovertemplate="%{y}<br>Δ %{x:.2f} п.п.<extra></extra>"))
    fig.add_vline(x=0, line_color="#ceab60", line_dash="dot")
    fig.update_xaxes(title="Δ P(владельцев ≥20 тыс.), п.п.")
    return fig
