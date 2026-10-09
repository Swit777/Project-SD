from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .dataset import GAME_GENRES

GENERIC_TAGS = {g.casefold() for g in GAME_GENRES} | {"early access", "free to play"}
FORMAT_TAGS = ["2D", "3D", "Pixel Graphics", "First-Person", "Third Person", "Top-Down", "Isometric",
               "Open World", "Sandbox", "Story Rich", "Horror", "Souls-like", "Roguelike", "Roguelite",
               "Action RPG", "JRPG", "Turn-Based", "Metroidvania", "Colony Sim", "Automation", "Farming Sim"]


def tag_tokens(value):
    try:
        tags = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return []
    return sorted({str(t).casefold() for t in tags if str(t).casefold() not in GENERIC_TAGS}) if isinstance(tags, list) else []


class TagIndex:
    """Unsupervised metadata retrieval index; no reviews, owner labels or model targets."""

    def __init__(self, catalog):
        self.ids = catalog.appid.to_numpy()
        self.vectorizer = TfidfVectorizer(analyzer=tag_tokens, norm="l2")
        documents = catalog.get("tags_json", pd.Series("[]", index=catalog.index)).fillna("[]")
        self.matrix = self.vectorizer.fit_transform(documents) if any(documents.map(tag_tokens)) else None

    def scores(self, tags):
        if self.matrix is None:
            return None
        query = self.vectorizer.transform([json.dumps(tags)])
        if not query.nnz:
            return None
        values = np.asarray((self.matrix @ query.T).toarray()).ravel()
        return pd.Series(values, index=self.ids)
