from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


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


def compute_classwise_metrics(pred_labels: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    p_real = precision_score(targets, pred_labels, pos_label=0, zero_division=0)
    r_real = recall_score(targets, pred_labels, pos_label=0, zero_division=0)
    f1_real = f1_score(targets, pred_labels, pos_label=0, zero_division=0)
    p_fake = precision_score(targets, pred_labels, pos_label=1, zero_division=0)
    r_fake = recall_score(targets, pred_labels, pos_label=1, zero_division=0)
    f1_fake = f1_score(targets, pred_labels, pos_label=1, zero_division=0)
    return {
        "precision_real": float(p_real),
        "recall_real": float(r_real),
        "f1_real": float(f1_real),
        "precision_fake": float(p_fake),
        "recall_fake": float(r_fake),
        "f1_fake": float(f1_fake),
    }


def compute_all_metrics(pred_labels: np.ndarray, targets: np.ndarray, pred_scores: np.ndarray | None = None) -> Dict[str, float | list]:
    score_vec = pred_scores if pred_scores is not None else pred_labels
    tn, fp, fn, tp = confusion_matrix(targets, pred_labels, labels=[0, 1]).ravel()
    out: Dict[str, float | list] = {
        "accuracy": compute_accuracy(pred_labels, targets),
        "precision": compute_precision(pred_labels, targets),
        "recall": compute_recall(pred_labels, targets),
        "f1_score": compute_f1_score(pred_labels, targets),
        "auc_roc": compute_auc_roc(score_vec, targets),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    out.update(compute_classwise_metrics(pred_labels, targets))
    return out
