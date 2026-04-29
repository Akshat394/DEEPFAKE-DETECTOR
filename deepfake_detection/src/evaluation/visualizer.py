from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_roc_curve(predictions: np.ndarray, targets: np.ndarray, save_path: str = "roc_curve.png"):
    fpr, tpr, _ = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(predictions: np.ndarray, targets: np.ndarray, save_path: str = "confusion_matrix.png"):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(history: Dict[str, List], save_path: str):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.get("train_loss", []), label="Train")
    plt.plot(history.get("val_loss", []), label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.get("train_acc", []), label="Train")
    plt.plot(history.get("val_acc", []), label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
