from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import TARGET, feature_sets
from .source import ARTICLE_URL, SOURCE_DOI


VERDICTS = {
    "supported": "Гипотеза поддержана на данной выборке: временные признаки снизили ошибку.",
    "dynamic_worse": "Гипотеза не поддержана: временные признаки увеличили ошибку.",
    "inconclusive": "Убедительных свидетельств преимущества временных признаков не получено.",
}


def write_reports(data: pd.DataFrame, result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    metadata = result["metadata"]
    hypothesis = metadata["hypothesis"]
    metrics = result["metrics"].sort_values("validation_player_mae")
    predictions = result["predictions"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 140})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#169978" if name == metadata["best_model"] else "#65747c" for name in metrics["model"]]
    ax.barh(metrics["model"], metrics["player_mae"], color=colors)
    ax.set(xlabel="MAE по игрокам, шкала 0–100 (меньше лучше)", title="Проверка на новых игроках")
    ax.invert_yaxis()
    save(fig, "model_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(data[TARGET], bins=np.linspace(0, 100, 26), color="#169978", edgecolor="white")
    ax.set(xlabel="Самооценка удовольствия, 0–100", ylabel="Число ответов", title="Распределение целевой переменной")
    save(fig, "target_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 2.4))
    delta = hypothesis["delta_mae"]
    low, high = hypothesis["ci_low"], hypothesis["ci_high"]
    ax.plot([low, high], [0, 0], color="#169978", linewidth=4)
    ax.scatter([delta], [0], color="#23282e", s=60, zorder=3)
    ax.axvline(0, color="#bb5544", linestyle="--")
    ax.set(yticks=[], xlabel="MAE aggregate − MAE dynamic; положительное значение лучше для dynamic",
           title="Парный bootstrap по игрокам: 95% интервал")
    save(fig, "hypothesis_interval.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hexbin(predictions[TARGET], predictions["prediction"], gridsize=28, mincnt=1, cmap="Greens")
    ax.plot([0, 100], [0, 100], "--", color="#bb5544")
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Фактическая оценка", ylabel="Оценка модели",
           title="Предсказания выбранной модели на test")
    save(fig, "predicted_vs_actual.png")

    importance = result["importance"].head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance["feature"], importance["mae_increase"], color="#667780")
    ax.axvline(0, color="#bb5544", linewidth=1)
    ax.set(xlabel="Рост MAE при перестановке признака", title="Permutation importance: RF dynamic")
    save(fig, "feature_importance.png")

    test = metadata["test_metrics"][metadata["best_model"]]
    splits = metadata["splits"]
    windows_text = ", ".join(str(w) for w in metadata["windows_minutes"])
    is_original = "preparation_audit" in metadata
    data_description = (
        f"Источник: [PowerWash Simulator Research Dataset]({SOURCE_DOI}), лицензия CC0-1.0.\n"
        f"[Статья создателей набора]({ARTICLE_URL}). Это реальные наблюдения, не синтетическая симуляция.\n"
        "Текущий эксперимент использует воспроизводимую подвыборку, а не весь архив.\n"
        "Цель Enjoyment переводится из 0–1000 в 0–100 делением на 10.\n"
        "Допускаются игроки с не менее чем тремя исходными валидными ответами, случайно выбираются игроки\n"
        "и ответы по лимитам конфигурации. После фильтрации эпизодов ответов может быть меньше трёх.\n"
        f"Из {metadata['preparation_audit']['input_responses']} выбранных ответов исключены "
        f"{metadata['preparation_audit']['missing_episode']} без активного эпизода и "
        f"{metadata['preparation_audit']['episode_context_mismatch']} с несовпадением контекста.\n"
    ) if is_original else (
        "Вход: пользовательская подготовленная таблица с признаками и целью в шкале 0–100.\n"
        "Её происхождение и лицензия не проверены автоматически. PowerWash упоминается как референсный\n"
        f"набор для протокола: [описание]({ARTICLE_URL}), [данные]({SOURCE_DOI}).\n"
        "Описанная ниже реконструкция эпизодов выполнена только для исходного PowerWash-набора.\n"
        "Для другого CSV она является требованием к семантике входных признаков, а не проверенным фактом.\n"
    )
    table = "\n".join(f"| {row.model} | {row.validation_player_mae:.3f} | {row.player_mae:.3f} | {row.rmse:.3f} | {row.r2:.3f} |"
                      for row in metrics.itertuples())
    text = f"""# Оценка субъективного игрового опыта по динамике поведения игрока

Рабочий исследовательский отчёт. Числа сформированы автоматически из текущего эксперимента.

## Аннотация

Исследуется возможность оценки самоотчёта Enjoyment по телеметрии PowerWash Simulator.
На {len(data):,} наблюдениях от {data.player_id.nunique():,} игроков сравниваются постоянный прогноз,
Ridge, Random Forest и Bayesian Ridge. Главный эксперимент сравнивает одинаковый Random Forest
с агрегированными признаками и с добавленной историей действий за {windows_text} минут.
{VERDICTS[hypothesis['verdict']]}
Разность ошибок составляет {delta:.3f} пункта; 95% bootstrap-интервал [{low:.3f}; {high:.3f}].

## Вопрос и гипотеза

Даёт ли недавняя динамика выполнения игровых подзадач дополнительную информацию о субъективном
удовольствии по сравнению с контекстом уровня и общей активностью в текущем эпизоде?
H1: MAE модели с временными признаками ниже MAE модели с агрегированными признаками на новых игроках.
Основная величина: среднее по игрокам их индивидуальных MAE, а не среднее по всем ответам.
Положительная разность aggregate − dynamic означает преимущество динамики.

## Данные и подготовка

{data_description}

Признаки строятся только по событиям строго до ответа. События с одинаковой с ответом секундой исключены.
Границы job_started/job_resumed начинают эпизод; job_exited/exited_game/player_logged_in прерывают его.
Ответы без подтверждённого активного эпизода или с несовпадающим уровнем/режимом исключаются.
История не переносится между эпизодами. Используются темп завершения подзадач, паузы, вариативность
интервалов, прирост прогресса между первым и последним наблюдаемыми событиями и покрытие временного окна.
ID игрока, ID ответа и время ответа не подаются модели.

## Экспериментальный протокол

- Train: {splits['train']['players']} игроков, {splits['train']['observations']} ответов.
- Validation: {splits['validation']['players']} игроков, {splits['validation']['observations']} ответов.
- Test: {splits['test']['players']} игроков, {splits['test']['observations']} ответов.
- Игроки между частями не пересекаются; параметры обработки обучаются только на train.
- Каждый игрок получает одинаковый суммарный вес при обучении; лучшие модели выбираются по validation.
- Дополнительная GroupKFold-проверка выполняется только на train + validation.
- Основная проверка гипотезы: парный bootstrap по {hypothesis['players']} test-игрокам, {hypothesis['bootstrap_repeats']} повторов.
- Сравнение окон и permutation importance являются дополнительным, разведочным анализом.

## Результаты

| Модель | Validation MAE по игрокам | Test MAE по игрокам | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
{table}

Выбрана **{metadata['best_model']}** исключительно по validation. Test MAE по игрокам: {test['player_mae']:.3f}.
{VERDICTS[hypothesis['verdict']]} Это вывод об указанном протоколе, а не доказательство невозможности
предсказания игрового опыта любыми другими методами.

## Неопределённость и объяснение

Дескриптивный интервал вокруг прогноза использует 90-й процентиль абсолютной ошибки на validation.
Радиус: {metadata['interval']['radius']:.2f}; доля попаданий на test: {metadata['interval']['test_coverage']:.1%}.
Повторные ответы зависимы, поэтому гарантии 90% покрытия нет. Такой интервал нельзя считать
вероятностью конкретного эмоционального состояния. Bayesian Ridge служит отдельной вероятностной моделью.
Permutation importance показывает чувствительность прогнозов к перестановке входов; коррелирующие
признаки могут делить важность. Это не причинное объяснение и не SHAP/LIME.

## Ограничения

Одна игра, добровольцы исследовательской версии, субъективные ответы и преобладание высоких оценок.
Фильтрация эпизодов исключает существенную часть ответов и может создавать селекционное смещение;
точные числа исключений находятся в аудите подготовки исходного набора.
Сведения о смене эпизодов могут быть неполными; внутри длинных эпизодов нельзя уверенно отличить паузу
в игре от отсутствующей телеметрии. Предсказание относится к Enjoyment, а не автоматически к скуке,
усталости или фрустрации. Для оценки будущих состояний нужен отдельный временной протокол.
Эффективность динамической адаптации игры здесь не проверяется: для неё нужен интервенционный эксперимент.
Поддерживается другой CSV с той же семантикой признаков; перенос на другие игры не гарантирован.

## Собственный вклад и воспроизводимость

Собственный вклад проекта: определение вопроса, реконструкция эпизодов, извлечение прошлых окон,
контроль утечек, независимая проверка на игроках, парное сравнение моделей, bootstrap и анализ ограничений.
Алгоритмы ML используются из проверенных библиотек, но исследование не сводится к одному вызову fit.
Глобальная новизна гипотезы не заявляется без отдельного обзора литературы.
Для повторения: `python run_pipeline.py run-all`. Настройки, версии библиотек, seed,
SHA-256 входного CSV и исходного кода сохранены в `run_manifest.json`.
Изменение данных означает новые численные результаты, но тот же вычислительный протокол.
"""
    (output / "research_report_ru.md").write_text(text, encoding="utf-8")
    schema = pd.DataFrame({"column": data.columns, "dtype": [str(data[c].dtype) for c in data],
                           "missing_fraction": [data[c].isna().mean() for c in data]})
    schema.to_csv(output / "dataset_summary.csv", index=False)
    sets = feature_sets(metadata["windows_minutes"])
    selected = data[data.player_id.isin(predictions.player_id)].head(50)
    selected[sets["dynamic"]].to_csv(output / "example_prediction_input.csv", index=False)
    selected["response_id"].to_csv(output / "example_prediction_ids.csv", index=False)
