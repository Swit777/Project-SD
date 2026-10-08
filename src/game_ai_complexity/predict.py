from __future__ import annotations

from pathlib import Path
import pickle

import pandas as pd

from .dataset import INFERENCE_COLUMNS, validate_inference_dataset
from .models import MODEL_INPUT_COLUMNS


def load_model(model_path: str | Path):
    with Path(model_path).open("rb") as fh:
        return pickle.load(fh)


def predict_success(data: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    errors = validate_inference_dataset(data)
    if errors:
        raise ValueError("; ".join(errors))

    model = load_model(model_path)
    probabilities = model.predict_proba(data[MODEL_INPUT_COLUMNS])[:, 1]
    result = data[INFERENCE_COLUMNS].copy()
    result["predicted_success_probability"] = probabilities
    result["predicted_success"] = (probabilities >= 0.5).astype(int)
    result["predicted_difficulty"] = 1.0 - probabilities
    return result

