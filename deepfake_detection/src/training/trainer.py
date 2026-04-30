from __future__ import annotations

import os
from typing import Dict

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class ModelTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        early_stopping_patience: int = 10,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True,
        use_amp_eval: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
        self.use_amp = bool(use_amp)
        self.use_amp_eval = bool(use_amp_eval)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        use_amp = self.device.type == "cuda" and self.use_amp
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        self.optimizer.zero_grad(set_to_none=True)
        for step_idx, (pixel_input, ascii_input, labels) in enumerate(train_loader):
            pixel_input = pixel_input.to(self.device)
            ascii_input = ascii_input.to(self.device)
            labels = labels.float().to(self.device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self.model(pixel_input, ascii_input)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                loss = self.criterion(logits.float(), labels.float(), model_params=list(self.model.parameters()))
                loss = loss / self.gradient_accumulation_steps
            scaler.scale(loss).backward()

            should_step = ((step_idx + 1) % self.gradient_accumulation_steps == 0) or ((step_idx + 1) == len(train_loader))
            if should_step:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.gradient_accumulation_steps
            probs = torch.sigmoid(logits)
            pred_labels = (probs > 0.5).float()
            correct += int((pred_labels == labels).sum().item())
            total += int(labels.numel())

        return {"loss": total_loss / max(len(train_loader), 1), "accuracy": correct / max(total, 1)}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        use_amp = self.device.type == "cuda" and self.use_amp_eval
        invalid_batches = 0

        for pixel_input, ascii_input, labels in val_loader:
            pixel_input = pixel_input.to(self.device)
            ascii_input = ascii_input.to(self.device)
            labels = labels.float().to(self.device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self.model(pixel_input, ascii_input)
            if not torch.isfinite(logits).all():
                invalid_batches += 1
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            loss = self.criterion(logits.float(), labels.float(), model_params=list(self.model.parameters()))

            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            pred_labels = (probs > 0.5).float()
            correct += int((pred_labels == labels).sum().item())
            total += int(labels.numel())

            all_preds.extend(pred_labels.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        if invalid_batches > 0:
            print(f"Validation warning: non-finite logits detected in {invalid_batches} batch(es); sanitized with nan_to_num.")

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
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            # Backward compatibility for older PyTorch versions without weights_only.
            ckpt = torch.load(path, map_location=self.device)
        load_result = self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if load_result.missing_keys:
            print(f"Checkpoint load: missing model keys={len(load_result.missing_keys)} (expected after architecture updates)")
        if load_result.unexpected_keys:
            print(f"Checkpoint load: unexpected model keys={len(load_result.unexpected_keys)}")

        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as exc:
            print(f"Checkpoint load: optimizer state skipped ({exc})")
        try:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as exc:
            print(f"Checkpoint load: scheduler state skipped ({exc})")
        self.best_val_loss = float(ckpt.get("val_loss", float("inf")))
        return int(ckpt.get("epoch", 0)) + 1

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_path: str = "checkpoints/",
        start_epoch: int = 0,
    ) -> Dict:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
        print(f"Starting training for {epochs} epochs on device={self.device} (start_epoch={start_epoch + 1})")

        for epoch in range(start_epoch, epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} | "
                f"lr={current_lr:.6e}"
            )

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics["loss"], save_path)
                print(f"  -> New best val_loss. Checkpoint saved to {os.path.join(save_path, 'best_model.pt')}")
            else:
                self.patience_counter += 1
                print(f"  -> No improvement. Early-stop patience: {self.patience_counter}/{self.early_stopping_patience}")
                if self.patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        print("Training finished.")
        return history
