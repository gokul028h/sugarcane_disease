import optuna
import torch
import torch.optim as optim
import yaml
from pathlib import Path
from src.models.model_factory import get_model
from src.models.losses import FocalLoss
from src.data.data_loaders import get_advanced_dataloaders
from src.training.trainer import RobustTrainer

def objective(trial):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Optuna suggests hyperparameters
    lr = trial.suggest_float("lr", config["optuna"]["lr_min"], config["optuna"]["lr_max"], log=True)
    batch_size = trial.suggest_categorical("batch_size", config["optuna"]["batch_sizes"])
    
    NUM_CLASSES = config["num_classes"]
    IMG_SIZE = config["input_size"]
    EPOCHS = min(10, config["epochs"]) # Shorter tuning epochs
    ADV_CFG = config["advanced_training"]
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, _, details = get_advanced_dataloaders(batch_size, IMG_SIZE, use_class_weights=True)
    
    model = get_model("efficientnet", NUM_CLASSES, pretrained=True)
    
    class_weights = details["class_weights"].to(DEVICE) if details["class_weights"] is not None else None
    criterion = FocalLoss(alpha=class_weights, gamma=ADV_CFG["focal_loss_gamma"])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Simple scheduler for tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    exp_dir = Path("experiments") / "optuna_trials" / f"trial_{trial.number}"
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    trainer = RobustTrainer(
        model=model,
        device=DEVICE,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=3, # aggressive early stopping for tuning
        exp_dir=str(exp_dir),
        mixup_cutmix_fn=None # disabled for faster stable tuning
    )

    metrics = trainer.fit(train_loader, val_loader, EPOCHS)
    return metrics["best_val_accuracy"]

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config["optuna"]["n_trials"])

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
