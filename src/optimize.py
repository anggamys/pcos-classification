import optuna
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .train import train_one_epoch, validate


def objective(trial, model_class, train_loader, val_loader, device, config_base):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSProp", "SGD"])

    model = model_class(num_classes=1, pretrained=True, dropout=dropout)
    model = model.to(device)

    if optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSProp":
        optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0
    max_epochs = config_base.get("epochs", 20)
    patience = config_base.get("patience", 5)

    for epoch in range(max_epochs):
        train_loss, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def run_optuna(model_class, train_loader, val_loader, device, config_base,
               n_trials=20, study_name="pcos_optuna"):
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    study.optimize(
        lambda trial: objective(
            trial, model_class, train_loader, val_loader, device, config_base
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\nBest trial:")
    print(f"  Value (val_loss): {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial.params
