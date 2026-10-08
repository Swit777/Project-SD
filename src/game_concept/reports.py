from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import BAND_LABELS, MECHANIC_COLUMNS, feature_columns
from .mechanics import LABELS

VERDICTS = {"supported": "Взаимодействия дали статистически различимое улучшение на выбранном test-наборе.",
            "interactions_worse": "Модель взаимодействий оказалась хуже аддитивной на выбранном test-наборе.",
            "inconclusive": "Убедительных свидетельств преимущества взаимодействий в текущем протоколе нет."}


def write_reports(data: pd.DataFrame, result: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 140})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, bbox_inches="tight")
        plt.close(fig)
    market = result["market_metrics"].sort_values("validation_group_log_loss")
    meta, hypothesis = result["metadata"], result["metadata"]["hypothesis"]
    fig, ax = plt.subplots(figsize=(8, 3.7))
    ax.barh(market.model, market.group_log_loss, color=["#179b78" if n == meta["selected_model"] else "#74828b" for n in market.model])
    ax.invert_yaxis()
    ax.set(xlabel="Test log loss по разработчикам (меньше лучше)", title="Прогноз оценочных диапазонов владельцев")
    save(fig, "model_comparison.png")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = data.owners_band.value_counts().reindex(range(4), fill_value=0)
    ax.bar(BAND_LABELS, counts, color="#74828b")
    ax.set(xlabel="Оценочный диапазон SteamSpy", ylabel="Игры", title="Распределение масштаба аудитории")
    save(fig, "owner_distribution.png")
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.plot([hypothesis["ci_low"], hypothesis["ci_high"]], [0, 0], color="#179b78", linewidth=4)
    ax.scatter([hypothesis["delta_log_loss"]], [0], color="#22262d", s=60, zorder=3)
    ax.axvline(0, linestyle="--", color="#ba5b52")
    ax.set(yticks=[], xlabel="Log loss additive − interactions", title="95% парный bootstrap по разработчикам")
    save(fig, "interaction_hypothesis.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = data[MECHANIC_COLUMNS].sum().sort_values(ascending=False).head(12).iloc[::-1]
    ax.barh([LABELS[c.removeprefix("mechanic_")] for c in counts.index], counts, color="#ceab60")
    ax.set(xlabel="Игры с найденным свидетельством", title="Извлечённые игровые элементы")
    save(fig, "mechanic_coverage.png")
    curves = result["reliability"]
    diagnostics = result["calibration_metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, threshold in zip(axes, [20000, 100000, 500000]):
        for stage, label, color in [("before", "До", "#74828b"), ("after", "После", "#179b78")]:
            points = curves[curves.model.eq(meta["selected_model"]) & curves.stage.eq(stage) & curves.threshold.eq(threshold) & curves.games.gt(0)]
            ax.plot(points.mean_probability, points.observed_fraction, "o-", color=color, label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#aaaaaa")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Средний прогноз", ylabel="Наблюдаемая доля", title=f"Аудитория ≥ {threshold:,}")
        ax.legend()
    save(fig, "probability_calibration.png")
    chosen_diagnostics = diagnostics[diagnostics.model.eq(meta["selected_model"])]
    calibration_table = "\n".join(f"| {int(r.threshold):,} | {r.stage} | {r.ece:.4f} | {r.binary_brier:.4f} | {r.group_log_loss:.4f} |" for r in chosen_diagnostics.itertuples())
    input_schema = {"feature_columns": feature_columns(), "mechanic_flags": MECHANIC_COLUMNS,
                    "training_extra_columns": ["appid", "name", "developer_group", "release_date", "owners_lower", "owners_upper", "owners_band", "playtime_median_hours"],
                    "owner_band_labels": BAND_LABELS, "is_actual_sales": False, "is_retention": False}
    (output / "input_schema.json").write_text(json.dumps(input_schema, indent=2), encoding="utf-8")
    evidence_path = Path(__file__).resolve().parents[2] / "data/processed/game_concept/mechanic_evidence.csv"
    if meta["input_origin"] == "pinned Steam snapshot" and evidence_path.exists():
        evidence = pd.read_csv(evidence_path)
        positive = evidence[~evidence.negated.astype(str).str.lower().eq("true")]
        annotation = positive.groupby(["mechanic", "source_type"], group_keys=False).sample(
            n=1, random_state=meta["config"]["seed"])
        annotation = annotation.drop_duplicates(["appid", "mechanic", "source_type"]).copy()
        annotation["expert_confirmed"] = ""
        annotation["expert_notes"] = ""
        annotation.to_csv(output / "mechanic_annotation_template.csv", index=False)
    split_table = "\n".join(f"| {name} | {values['games']} | {values['developer_groups']} | {values['playtime_available']} |" for name, values in meta["splits"].items())
    metric_table = "\n".join(f"| {r.model} | {r.validation_group_log_loss:.4f} | {r.group_log_loss:.4f} | {r.brier:.4f} | {r.balanced_accuracy:.3f} |" for r in market.itertuples())
    playtime = meta["playtime"]
    playtime_text = f"Выбрана {playtime.get('selected_model')}; фактическое test-покрытие интервала: {playtime.get('empirical_test_coverage', 0):.1%}." if playtime["status"] == "available" else "Недостаточно данных для модели игрового времени."
    if meta["input_origin"] == "pinned Steam snapshot":
        source_text = f"""[Steam Games Dataset / FronkonGames](https://huggingface.co/datasets/FronkonGames/steam-games-dataset), ревизия `{meta['snapshot_revision']}`.
Карточка указывает MIT для набора; права на сторонние описания, изображения и товарные знаки сохраняются.
Данные подготовлены из снимка Steam/SteamSpy; прямой парсер публичных страниц Steam выполняет независимый аудит.
Исходные тексты и HTTP-ответы хранятся локально, без обязательной публикации в GitHub.
Аудит прямого парсинга и дата каждого ответа находятся в `data/processed/game_concept/scrape_manifest.json` и HTTP-кеше.
Источники не смешиваются: текущие live-механики не соединяются автоматически со старыми метками результата.

В эксперименте {len(data):,} платных игр со свидетельствами описания, валидными owner-бандами и возрастом согласно конфигурации.
Случайная воспроизводимая выборка по хешу AppID+seed, без отбора только коммерческих победителей.
Исходный CSV имел соединённый заголовок Discount/DLC count; исправление явно записано в аудите, значения не сдвигаются."""
    else:
        source_text = f"""Пользовательский набор: {len(data):,} строк; SHA-256 `{meta['input_sha256']}`.
Происхождение, лицензия, способ отбора и значение целевых меток требуют проверки поставщиком.
Проверена структура, но происхождение Steam не подтверждено. Описания и теги автоматически не извлекались.
Модель трактует метки по заявленной схеме owner-бандов и часов; это должно соответствовать смыслу загруженных данных."""
    text = f"""# Прогнозирование коммерческого потенциала игровой концепции по сочетаниям механик

Рабочий отчёт первого эксперимента. Коммерческий результат в этом наборе — **оценочный диапазон владельцев**, не фактические продажи.

## Вопрос и гипотеза

Улучшают ли взаимодействия игровых элементов прогноз коммерческого потенциала относительно независимых вкладов компонентов?
Основное сравнение: одинаковый HistGradientBoostingClassifier с запретом взаимодействий (`no_interactions`) и без этого ограничения.
Контекст, признаки, число итераций, регуляризация, выборки обучения и калибровки совпадают.
Отдельно сравниваются константный prior, модель жанров/контекста и Logistic Regression.
Цель — оценка концепции на основе аналогов, не доказательство причинного эффекта добавления механики.

## Источник и извлечение данных

{source_text}

## Игровые элементы

Словарь версии `{meta['taxonomy_version']}` содержит {len(MECHANIC_COLUMNS)} элементов: крафт, строительство базы, процедурная генерация,
постоянная смерть, кооператив, развитие навыков, ветвящийся сюжет и другие.
Извлечение использует структурированные функции Steam, точные теги и узкие фразы описания разработчика.
Каждое свидетельство сохраняет AppID, URL, источник, текст и признак отрицания.
Публикация механики в описании не подтверждает качество её реализации. Отсутствие свидетельства не доказывает отсутствие механики.
Дополнительно сохраняются кандидаты особенных возможностей (`feature_claims.csv`), которые требуют ручной разметки;
они не становятся автоматически входами модели и не получают выдуманных оценок «интересности».
Точность извлечения требует независимой ручной проверки: готовая форма аудита не является измеренной precision/recall.
Добавлен полный аудит всех элементов на случайно отобранных играх: `audit-create --games 40`.
Пропуски разметки не становятся отрицательными; `unclear` исключается из метрик с явным счётчиком.
Сохранённые экспертные метки не изменяют автоматически обучение. Измеряется подтверждение утверждений метаданными,
а не проверенное наличие/качество игровых механик. Неполная разметка не устанавливает точность всего корпуса.

## Контроль утечек и оценка

Входы: жанры, признаки элементов, их число, восстановленная недисконтная цена снимка, возраст аналогов,
число языков и платформы. Цена снимка не гарантированно равна стартовой цене.
Отзывы, оценки, CCU, число владельцев и игровые часы **не входят** в прогнозирующие признаки.
Игры с любым общим разработчиком объединяются в связную группу; группы между частями не пересекаются.
Франшизы разных студий и издательские связи дополнительно могут давать зависимость, которую эта группировка не полностью снимает.

| Часть | Игры | Группы разработчиков | Доступные часы |
|---|---:|---:|---:|
{split_table}

Обучение взвешивается по группам разработчиков. Лучшая модель выбирается по validation, независимо от test.
Вероятности неконстантных моделей калибруются sigmoid-методом на отдельной calibration-части, без дообучения базовой модели.
Калибровка использует равный вес игр. Test log loss дополнительно усредняется сначала в группе, затем между группами.
Парный bootstrap пересэмплирует группы разработчиков. Он отражает неопределённость test-сравнения, а не всех возможных обучающих выборок.
Также проведён отдельный некалиброванный временной backtest: более новые релизы отложены,
старые игры их разработчиков исключены из обучения. Это не проверка настоящего pre-launch-прогноза:
текущие теги, описания и цены могут изменяться после релиза.

## Результаты

| Модель | Validation group log loss | Test group log loss | Test Brier | Test balanced accuracy |
|---|---:|---:|---:|---:|
{metric_table}

Выбрана по validation **{meta['selected_model']}**.
Основная разность ошибок: {hypothesis['delta_log_loss']:+.4f}; 95% интервал [{hypothesis['ci_low']:+.4f}; {hypothesis['ci_high']:+.4f}].
{VERDICTS[hypothesis['verdict']]}
Сложность модели не является основанием объявить её лучше. При сильном дисбалансе высокая общая accuracy может быть тривиальной;
поэтому показываются proper scoring rules, balanced accuracy и сравнение с prior.

## Проверка вероятностей

Для выбранной по validation модели — отдельные test-диагностики до/после калибровки.
Пороги 20k, 100k и 500k, десять одинаковых интервалов вероятности.
ECE — взвешенная по числу игр средняя абсолютная разность среднего прогноза и наблюдаемой доли в интервале.
Binary Brier — средний квадрат ошибки вероятности для порогового события.

| Граница | Этап | ECE | Binary Brier | Group multiclass log loss |
|---|---|---:|---:|---:|
{calibration_table}

`figures/probability_calibration.png`, `reliability_bins.csv` сохраняют кривые и размеры групп.
Диагностики усреднены по играм, log loss — по разработчикам; это разные целевые распределения.
ECE чувствителен к выбору интервалов и малым частотам; у кривой нет независимой гарантии точности нового продукта.
Калибровка может ухудшить отдельный показатель, что нельзя скрывать. Диагностики не используются для выбора модели.

## Вовлечённость и объяснение

Для игрового времени сравниваются медиана, Bayesian Ridge и HGB на `log(1+hours)`.
Нулевые/невалидные исходные часы не трактуются как измеренное отсутствие интереса; прогноз обучен только на доступных положительных значениях.
{playtime_text} Интервал калибруется по абсолютным остаткам отдельной calibration-части; при зависимых наблюдениях и сдвиге домена гарантии покрытия нет.
Это не возврат на D7/D30. Отдельная свежая выборка времени у авторов отзывов тоже смещена самоотбором и не является retention.
Permutation importance объясняет чувствительность полной модели; LIME даёт локальную аппроксимацию для диапазона 20k–100k.
В permutation importance число механик пересчитывается при каждой перестановке и не трактуется как независимый вход.
Базовая и изменённая ошибки сравниваются на одной и той же seed-подвыборке до 1 200 test-игр.
Это сохраняет производный признак, но не гарантирует поддержку всех полученных сочетаний в реальном корпусе.
Низкий local R² означает ненадёжную аппроксимацию. Ни один метод не доказывает причинный вклад механики.

## Конструктор концепции

Прогноз включает распределение owner-бандов, вероятность превышения их известных границ, оценку часов,
похожие **train-игры** и сравнение изменения компонентов при фиксированном контексте.
Сравнение включает межмодельное расхождение Logistic / аддитивной HGB / HGB с взаимодействиями:
при разных направлениях изменения компонент не представляется надёжной рекомендацией.
Разброс моделей не является доверительным интервалом. Порог ±0.25 п.п. — обозначение близких к нулю изменений, не тест значимости.
Для слишком плотных, неизвестных или почти не встречавшихся комбинаций модель отказывается от численного прогноза.
Это защита от экстраполяции, а не искусственный штраф качеству и не доказательство того, что необычная игра обречена.
Короткая игра не считается плохой только из-за небольшого времени прохождения.
Возраст аналогов — условие сравнения накопленных показателей, не восстановленная кривая продаж будущего продукта.

## Ограничения и воспроизводимость

Нет фактических транзакций, рекламного бюджета, качества исполнения, возвратов и средних реализованных цен.
Owner-банда — шумная широкая оценка, а не копии продаж. Бесплатные игры, новые релизы и неподдерживаемые жанры вне этой постановки.
Временная проверка остаётся ретроспективной из-за post-release-метаданных. Перенос на другие платформы и рынок не проверен.
Полноценные годовые продажи, прибыль и cohort-retention требуют новых источников соответствующих меток.
Повторение: `python run_game_concept.py run-all`; прямые страницы повторяются из проверенного кеша,
а `scrape --refresh` явно создаёт новый сбор. В манифесте — seed, конфигурация, версии, SHA-256 данных и кода.
Для другого совместимого CSV повторяется методика, а не обещаются те же значения метрик.
"""
    (output / "research_report_ru.md").write_text(text, encoding="utf-8")
