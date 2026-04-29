from __future__ import annotations

import os
from typing import Dict

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class ModelTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, early_stopping_patience: int = 10):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for pixel_input, ascii_input, labels in train_loader:
            pixel_input = pixel_input.to(self.device)
            ascii_input = ascii_input.to(self.device)
            labels = labels.float().to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(pixel_input, ascii_input)
            loss = self.criterion(predictions, labels, model_params=list(self.model.parameters()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pred_labels = (predictions > 0.5).float()
            correct += int((pred_labels == labels).sum().item())
            total += int(labels.numel())

        return {"loss": total_loss / max(len(train_loader), 1), "accuracy": correct / max(total, 1)}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        for pixel_input, ascii_input, labels in val_loader:
            pixel_input = pixel_input.to(self.device)
            ascii_input = ascii_input.to(self.device)
            labels = labels.float().to(self.device)

            predictions = self.model(pixel_input, ascii_input)
            loss = self.criterion(predictions, labels, model_params=list(self.model.parameters()))

            total_loss += loss.item()
            pred_labels = (predictions > 0.5).float()
            correct += int((pred_labels == labels).sum().item())
            total += int(labels.numel())

            all_preds.extend(pred_labels.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        return {
            "loss": total_loss / max(len(val_loader), 1),
            "accuracy": correct / max(total, 1),
            "f1": float(f1_score(all_labels, all_preds, zero_division=0)) if all_labels else 0.0,
        }

    def save_checkpoint(self, epoch: int, val_loss: float, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            os.path.join(path, "best_model.pt"),
        )

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_loss = float(ckpt.get("val_loss", float("inf")))
        return int(ckpt.get("epoch", 0)) + 1

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50, save_path: str = "checkpoints/") -> Dict:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics["loss"], save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    break

        return history
