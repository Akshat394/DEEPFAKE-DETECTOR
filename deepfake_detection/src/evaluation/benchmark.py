from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .metrics import compute_all_metrics


class BenchmarkEvaluator:
    @torch.no_grad()
    def evaluate_on_dataset(self, model, dataset_loader, dataset_name: str):
        model.eval()
        preds, targets = [], []
        device = next(model.parameters()).device

        for pixel_input, ascii_input, labels in dataset_loader:
            pixel_input = pixel_input.to(device)
            ascii_input = ascii_input.to(device)
            scores = model(pixel_input, ascii_input)
            preds.extend((scores > 0.5).float().cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())

        metrics = compute_all_metrics(np.asarray(preds), np.asarray(targets))
        return {"dataset": dataset_name, **metrics}

    def cross_dataset_evaluation(self, model, train_dataset: str, test_dataset: str):
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "note": "Provide dataset loaders and call evaluate_on_dataset for full results.",
        }

    def ablation_study(self, base_model, ablation_config: Dict):
        return {
            "ablation_config": ablation_config,
            "note": "Implement per-experiment component toggles and rerun training/evaluation.",
        }
