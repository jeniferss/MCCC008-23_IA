import json
import logging
import re

import joblib
import numpy as np
import unicodedata

from app.core.config.settings import settings

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def predict_traits_snippet(
        snippet: str,
        model_path: str = settings.BASE_DIR / "data" / "modelo_traits.joblib",
        labels_path: str = settings.BASE_DIR / "data" / "traits_labels.json",
        thresholds_path: str = settings.BASE_DIR / "data" / "traits_thresholds.npy",
        top_k: int | None = None,
):
    model = joblib.load(model_path)
    thresholds = np.load(thresholds_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    snippet_clean = _normalize_text(snippet)
    probs = model.predict_proba([snippet_clean])[0]
    if top_k is not None:
        k = min(int(top_k), len(probs))
        idx = np.argsort(probs)[::-1][:k]
        selected_labels = [label_names[i] for i in idx]
        probs_by_label = {label_names[i]: float(probs[i]) for i in idx}

    else:
        idx = np.where(probs >= thresholds)[0]
        selected_labels = [label_names[i] for i in idx]
        probs_by_label = {label_names[i]: float(probs[i]) for i in idx}

    return {
        "labels": selected_labels,
        "scores": probs_by_label,
    }


def predict_traits_snippet2(
        snippet: str,
        max_k: int = 3,
        model_path: str = settings.BASE_DIR / "data" / "modelo_traits.joblib",
        labels_path: str = settings.BASE_DIR / "data" / "traits_labels.json",
        thresholds_path: str = settings.BASE_DIR / "data" / "traits_thresholds.npy",
):
    model = joblib.load(model_path)
    thresh = np.load(thresholds_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    if len(thresh) != len(label_names):
        thresh = np.array([0.5] * len(label_names))

    PENALTY = {
        "determinado": 0.6,
        "observador": 0.55,
        "gentil": 0.65
    }

    CUSTOM_THRESH = {
        "determinado": 0.28,
        "observador": 0.30,
        "gentil": 0.26
    }

    snippet_clean = _normalize_text(snippet)
    probs = model.predict_proba([snippet_clean])[0]

    adjusted_probs = []
    final_thresholds = []

    for i, lab in enumerate(label_names):
        p = float(probs[i])
        t = float(thresh[i])

        if lab in CUSTOM_THRESH:
            t = CUSTOM_THRESH[lab]

        if lab in PENALTY:
            p = p * PENALTY[lab]

        adjusted_probs.append(p)
        final_thresholds.append(t)

    adjusted_probs = np.array(adjusted_probs)
    final_thresholds = np.array(final_thresholds)

    selected = []
    for i, lab in enumerate(label_names):
        if adjusted_probs[i] >= final_thresholds[i]:
            selected.append((lab, float(adjusted_probs[i])))

    if len(selected) > max_k:
        selected = sorted(selected, key=lambda x: x[1], reverse=True)[:max_k]

    if len(selected) == 0:
        top_idx = np.argsort(adjusted_probs)[::-1][:max_k]
        selected = [(label_names[i], float(adjusted_probs[i])) for i in top_idx]

    return {
        "labels": [lab for lab, _ in selected],
        "scores": {lab: score for lab, score in selected}
    }
