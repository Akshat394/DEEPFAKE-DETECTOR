from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(accuracy_score(targets, predictions))


def compute_precision(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(precision_score(targets, predictions, zero_division=0))


def compute_recall(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(recall_score(targets, predictions, zero_division=0))


def compute_f1_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(f1_score(targets, predictions, zero_division=0))


def compute_auc_roc(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(roc_auc_score(targets, predictions))


def compute_all_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": compute_accuracy(predictions, targets),
        "precision": compute_precision(predictions, targets),
        "recall": compute_recall(predictions, targets),
        "f1_score": compute_f1_score(predictions, targets),
        "auc_roc": compute_auc_roc(predictions, targets),
    }
