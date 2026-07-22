from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {"acc": None, "prec_macro": None, "rec_macro": None, "f1_macro": None, "mcc": None, "n": 0}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "prec_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "rec_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else 0.0,
        "n": int(len(y_true)),
    }
