import json
import logging
import os
import re
import tempfile

import unicodedata

import requests

import joblib
import numpy as np

from core.config.settings import settings

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def predict_traits_snippet2(
        snippet: str,
        max_k: int = 3,
        model_path: str = settings.BASE_DIR / "data" / "modelo_traits.joblib",
        labels_path: str = settings.BASE_DIR / "data" / "traits_labels.json",
        thresholds_path: str = settings.BASE_DIR / "data" / "traits_thresholds.npy",
):
    url = "https://github.com/jeniferss/MCCC008-23_IA/raw/master/app/data/modelo_traits.joblib"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name
        model = joblib.load(tmp_path)

    os.unlink(tmp_path)

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
