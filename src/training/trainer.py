import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import json
from tqdm import tqdm
import time

class RobustTrainer:
    """
    A robust training loop with early stopping, TensorBoard logging,
    mixed-precision support, and per-epoch history tracking for
    IEEE publication-quality loss/accuracy curve generation.
    """
    def __init__(self, model, device, criterion, optimizer, scheduler, patience: int, exp_dir: str, mixup_cutmix_fn=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.exp_dir = Path(exp_dir)
        self.mixup_cutmix_fn = mixup_cutmix_fn
        
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.exp_dir / "logs"))
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        # Per-epoch history for plotting
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            if self.mixup_cutmix_fn is not None:
                images, labels = self.mixup_cutmix_fn(images, labels)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # For CutMix/MixUp, labels are soft, accuracy calculation is tricky during train
            if self.mixup_cutmix_fn is None:
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    def validate_epoch(self, dataloader, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader, epochs: int):
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate_epoch(val_loader, epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # TensorBoard logging
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            if train_acc > 0:
                self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)
            
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.exp_dir / "checkpoints" / "best_model.pth")
                print(">>> Saved new best model")
            else:
                self.epochs_without_improvement += 1
                
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        self.writer.close()
        
        total_time = time.time() - start_time
        
        # Count model parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Save metrics
        metrics = {
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_acc,
            "stopped_epoch": epoch,
            "total_training_time_seconds": round(total_time, 2),
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "history": self.history
        }
        with open(self.exp_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
