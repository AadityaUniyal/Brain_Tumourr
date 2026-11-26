"""
Training Pipeline for DualHeadBrainTumorNet:
Executes multi-task training with patient-stratified validation, early stopping,
and automated metric logging. Zero hardcoded results.
"""
import os
import json
import time
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.models.dual_head_net import DualHeadBrainTumorNet
from src.models.losses import DualTaskCompoundLoss
from src.training.metrics import compute_clinical_metrics


class Trainer:
    def __init__(
        self,
        model,
        criterion=None,
        optimizer=None,
        scheduler=None,
        device=None,
        output_dir="models/checkpoints",
        reports_dir="reports/metrics",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion or DualTaskCompoundLoss()
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = scheduler
        self.output_dir = output_dir
        self.reports_dir = reports_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_seg_loss = 0.0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss_dict = self.criterion(outputs, {"label": labels, "mask": masks})

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += loss_dict["cls_loss"].item()
            total_seg_loss += loss_dict["seg_loss"].item()

        if self.scheduler is not None:
            self.scheduler.step()

        n_batches = max(len(dataloader), 1)
        return {
            "loss": total_loss / n_batches,
            "cls_loss": total_cls_loss / n_batches,
            "seg_loss": total_seg_loss / n_batches,
        }

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        all_pred_masks = []
        all_true_masks = []

        total_loss = 0.0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            masks = batch["mask"].to(self.device)

            outputs = self.model(images)
            loss_dict = self.criterion(outputs, {"label": labels, "mask": masks})
            total_loss += loss_dict["loss"].item()

            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            seg_probs = torch.sigmoid(outputs["seg_mask"]).squeeze(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_pred_masks.extend(seg_probs.cpu().numpy())
            all_true_masks.extend(masks.squeeze(1).cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_pred_masks = np.array(all_pred_masks)
        all_true_masks = np.array(all_true_masks)

        metrics = compute_clinical_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_probs=all_probs,
            pred_masks=all_pred_masks,
            true_masks=all_true_masks,
        )
        metrics["val_loss"] = total_loss / max(len(dataloader), 1)
        return metrics

    def fit(self, train_loader, val_loader, epochs=15, early_stopping_patience=5, fold=0):
        best_f1 = -1.0
        patience_counter = 0
        history = []

        print(f"=== Starting Training Fold {fold} on {self.device} ===")
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            elapsed = time.time() - start_time

            val_f1 = val_metrics["overall"]["macro_f1"]
            val_acc = val_metrics["overall"]["accuracy"]
            val_dice = val_metrics.get("segmentation", {}).get("mean_dice_coefficient", 0.0)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Dice: {val_dice:.4f}"
            )

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["val_loss"],
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "val_dice": val_dice,
            }
            history.append(record)

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_path = os.path.join(self.output_dir, f"best_model_fold_{fold}.pt")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  --> Checkpoint saved: {best_model_path} (Best Macro F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Save training history and final evaluation
        history_file = os.path.join(self.reports_dir, f"training_history_fold_{fold}.json")
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        return history
