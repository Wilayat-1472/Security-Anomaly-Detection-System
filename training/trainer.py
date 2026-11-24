import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance and hard negatives in threat detection.
    Ref: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of true class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class Trainer:
    """
    Research-grade Trainer supporting:
    - Label smoothing & Focal loss
    - Mixed Precision Training (torch.amp)
    - Cosine Annealing with Warm Restarts
    - Class-balanced weighting
    - Gradient clipping
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        class_names: list,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.class_names = class_names

        train_cfg = config.get("training", {})
        loss_type = train_cfg.get("loss_type", "focal").lower()

        # Loss selection
        if loss_type == "focal":
            gamma = train_cfg.get("focal_gamma", 2.0)
            self.criterion = FocalLoss(alpha=class_weights, gamma=gamma)
        elif loss_type == "weighted_ce":
            label_smoothing = train_cfg.get("label_smoothing", 0.1)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer selection: AdamW with weight decay
        lr = train_cfg.get("learning_rate", 3e-4)
        weight_decay = train_cfg.get("weight_decay", 1e-4)
        opt_name = train_cfg.get("optimizer", "adamw").lower()

        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            momentum = train_cfg.get("momentum", 0.9)
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # LR Scheduler: CosineAnnealingLR or StepLR
        self.epochs = train_cfg.get("epochs", 30)
        sched_type = train_cfg.get("scheduler", "cosine").lower()
        if sched_type == "cosine":
            min_lr = train_cfg.get("min_lr", 1e-6)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=min_lr
            )
        else:
            step_size = train_cfg.get("scheduler_step", 7)
            gamma = train_cfg.get("scheduler_gamma", 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )

        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.use_amp = train_cfg.get("use_amp", False) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.patience = train_cfg.get("early_stop_patience", 12)
        self.save_dir = train_cfg.get("save_dir", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> tuple:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / max(total, 1):.4f}"})

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, val_loader: torch.utils.data.DataLoader) -> tuple:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, Any]:
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Training on device: {self.device} | Trainable parameters: {num_params:,}")
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<12}")

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            print(f"{epoch:<6} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {current_lr:<12.2e}")

            # Tracking best model based on validation loss / accuracy
            if val_loss < self.best_val_loss or val_acc > self.best_val_acc:
                if val_acc >= self.best_val_acc:
                    self.best_val_acc = val_acc
                self.best_val_loss = min(self.best_val_loss, val_loss)
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_acc, val_loss, is_best=True)
            else:
                self.patience_counter += 1

            if epoch % 5 == 0 or epoch == self.epochs:
                self._save_checkpoint(epoch, val_acc, val_loss, is_best=False)

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={self.patience})")
                break

        print(f"\nTraining complete. Peak Validation Accuracy: {self.best_val_acc:.4f}")
        return self.history

    def _save_checkpoint(self, epoch: int, val_acc: float, val_loss: float, is_best: bool = False):
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "class_names": self.class_names,
            "config": self.config,
        }, path)

    def load_best(self):
        path = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded optimal checkpoint (Epoch {checkpoint.get('epoch', '?')} - Val Acc: {checkpoint.get('val_acc', 0.0):.4f})")
            return checkpoint
        return None
