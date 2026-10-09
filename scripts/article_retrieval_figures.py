from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/game_concept/analogue_upgrade"


def make_figures(destination):
    destination.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((SOURCE / "comparison.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
    labels = ["RPG + крафт\n+ кооператив", "Стратегия + колода\n+ пошаговый бой", "Adventure + загадки\n+ исследование", "Action + платформинг\n+ боссы", "Simulation + база\n+ автоматизация"]
    fig, ax = plt.subplots(figsize=(10.5, 4.4), layout="constrained")
    for i, (key, title, color) in enumerate([("diagnostics", "Строгий", "#718895"),
                                            ("balanced_diagnostics", "Ближайшие", "#25836d"),
                                            ("broad_diagnostics", "Рыночные референсы", "#af7e28")]):
        values = [c[key]["eligible_games"] for c in comparison["cases"].values()]
        bars = ax.bar(np.arange(5) + (i - 1) * .26, values, width=.25, color=color, label=title)
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(range(5), labels)
    ax.set_ylabel("Число кандидатов после фильтров")
    ax.set_yscale("log")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    ax.margins(y=.25)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(destination / "09_retrieval_modes.png", dpi=200)
    plt.close(fig)

    anchor = pd.read_csv(SOURCE / "anchor_stardew.csv").head(8)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.barh(anchor.name.str.strip(), anchor.similarity, color="#25836d")
    ax.invert_yaxis()
    for i, row in enumerate(anchor.itertuples()):
        ax.text(row.similarity + 1, i, f"{row.review_count:,.0f} отзывов", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Гибридный балл сходства, 0–100; не вероятность успеха")
    ax.set_title("Ориентир: Stardew Valley; RPG, крафт и кооператив", fontsize=11)
    fig.savefig(destination / "10_tag_anchor.png", dpi=200)
    plt.close(fig)

    report = ROOT / "reports/game_concept"
    manifest = json.loads((report / "run_manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(report / "test_predictions.csv")
    scores = predictions[[f"{manifest['selected_model']}_p{k}" for k in range(4)]].to_numpy()
    matrix = confusion_matrix(predictions.owners_band, scores.argmax(axis=1), labels=range(4))
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.7), layout="constrained")
    im = ax.imshow(normalized, cmap="Greens", vmin=0, vmax=1)
    labels = ["<20 тыс.", "20–100 тыс.", "100–500 тыс.", "≥500 тыс."]
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(4), labels)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{matrix[i,j]}\n{normalized[i,j]:.1%}", ha="center", va="center",
                    color="white" if normalized[i,j] > .6 else "#25282b")
    ax.set_xlabel("Предсказанный диапазон")
    ax.set_ylabel("Диапазон источника")
    fig.colorbar(im, ax=ax, label="Доля внутри истинного класса")
    fig.savefig(destination / "07_class_errors.png", dpi=200)
    plt.close(fig)
    shutil.copy2(SOURCE / "ui/forecast_1440.png", destination / "08_application.png")
    shutil.copy2(SOURCE / "ui/anchor_1440.png", destination / "11_analogue_interface.png")
