"""
Brain Age Prediction from T2 MRI or cortical label maps

Training setup:
- Optimizer: Adam
- Loss: Smooth L1, Huber
- LR: 1e-3 with ReduceLROnPlateau scheduler
- Epochs: 300
- Batch size: 8
"""

import os
import csv
import time
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, ToTensor, ScaleIntensity
from monai.networks.nets import DenseNet201


# ------------------------------
# Configuration
# ------------------------------
class Config:
    train_dir = "/media/m-ssd5/Synthseg_BA_cs6/GitHub_BrainAge/Datasets/FETA_dHCP/T2_train"
    train_tsv = "/media/m-ssd5/Synthseg_BA_cs6/GitHub_BrainAge/Datasets/FETA_dHCP/train.tsv"

    test_dir = "/media/m-ssd5/Synthseg_BA_cs6/GitHub_BrainAge/Datasets/FETA_dHCP/T2_test"
    test_tsv = "/media/m-ssd5/Synthseg_BA_cs6/GitHub_BrainAge/Datasets/FETA_dHCP/test.tsv"

    batch_size = 8
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 300

    num_workers = 4
    pin_memory = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 43

    checkpoint_dir = "./Trained_models/Train_1"
    latest_model_name = "latest_model.pth"
    best_model_name = "best_model.pth"

    log_csv_name = "training_log.csv"
    curve_png_name = "training_curve.png"


# ------------------------------
# Reproducibility
# ------------------------------
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)
random.seed(Config.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.seed)


# ------------------------------
# Utilities
# ------------------------------
def load_ages_from_tsv(tsv_path):
    ages = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            age = row.get("scan_age_mri0gamri") or row.get("scan_age")
            if age is None or age == "":
                continue
            ages.append(float(age))
    return np.array(ages, dtype=np.float32)


def load_niftis(directory):
    files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])
    return files


def save_training_log(history, csv_path):
    """
    Save per epoch metrics to CSV.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_mae", "lr"])

        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["test_loss"][i],
                history["test_mae"][i],
                history["lr"][i],
            ])


def plot_training_curves(history, save_path):
    """
    Plot training curves and save to disk.
    Always plots train loss.
    Plots test loss and test MAE only if available.
    """
    epochs = history["epoch"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")

    if any(v is not None for v in history["test_loss"]):
        test_loss_vals = [np.nan if v is None else v for v in history["test_loss"]]
        plt.plot(epochs, test_loss_vals, label="Test Loss")

    if any(v is not None for v in history["test_mae"]):
        test_mae_vals = [np.nan if v is None else v for v in history["test_mae"]]
        plt.plot(epochs, test_mae_vals, label="Test MAE")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ------------------------------
# Dataset
# ------------------------------
class BrainAgeDataset(Dataset):
    def __init__(self, image_paths, ages):
        assert len(image_paths) == len(ages), (
            f"Mismatch between number of images ({len(image_paths)}) "
            f"and number of ages ({len(ages)})"
        )

        self.image_paths = image_paths
        self.ages = ages

        self.transform = Compose([
            ToTensor(),
            lambda x: x.unsqueeze(0),
            ScaleIntensity()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        age = self.ages[idx]

        img = self.transform(img)

        return img, torch.tensor(age, dtype=torch.float32)


# ------------------------------
# Model
# ------------------------------
class AgePredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = DenseNet201(
            spatial_dims=3,
            in_channels=1,
            out_channels=1
        )

    def forward(self, x):
        return self.densenet(x).squeeze()


# ------------------------------
# Evaluation
# ------------------------------
def evaluate(model, loader):
    model.eval()
    criterion = nn.SmoothL1Loss()

    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(Config.device, non_blocking=True)
            y = y.to(Config.device, non_blocking=True)

            out = model(x).view(-1)
            y = y.view(-1)

            loss = criterion(out, y)

            total_loss += loss.item() * x.size(0)
            preds.append(out.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    mae = np.mean(np.abs(preds - targets))
    total_loss /= len(loader.dataset)

    return total_loss, mae


# ------------------------------
# Training
# ------------------------------
def train():
    print("\nStarting training pipeline\n")
    print("Device:", Config.device)

    train_paths = load_niftis(Config.train_dir)
    train_ages = load_ages_from_tsv(Config.train_tsv)

    print(f"Train samples: {len(train_paths)}")

    if os.path.isdir(Config.test_dir) and os.path.isfile(Config.test_tsv):
        test_paths = load_niftis(Config.test_dir)
        test_ages = load_ages_from_tsv(Config.test_tsv)
        print(f"Test samples : {len(test_paths)}")
        test_dataset = BrainAgeDataset(test_paths, test_ages)
    else:
        print("Test set not found, training without test evaluation.")
        test_dataset = None

    train_dataset = BrainAgeDataset(train_paths, train_ages)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory
        )

    model = AgePredictionModel().to(Config.device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    criterion = nn.SmoothL1Loss()

    os.makedirs(Config.checkpoint_dir, exist_ok=True)

    log_csv_path = os.path.join(Config.checkpoint_dir, Config.log_csv_name)
    curve_png_path = os.path.join(Config.checkpoint_dir, Config.curve_png_name)

    best_mae = float("inf")

    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "test_mae": [],
        "lr": []
    }

    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for x, y in train_loader:
            x = x.to(Config.device, non_blocking=True)
            y = y.to(Config.device, non_blocking=True)

            optimizer.zero_grad()

            out = model(x).view(-1)
            y = y.view(-1)

            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        train_loss = epoch_loss / len(train_loader.dataset)

        if test_loader is not None:
            test_loss, test_mae = evaluate(model, test_loader)
            scheduler.step(test_loss)
        else:
            test_loss, test_mae = None, None
            scheduler.step(train_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - start

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_mae"].append(test_mae)
        history["lr"].append(current_lr)

        save_training_log(history, log_csv_path)
        plot_training_curves(history, curve_png_path)

        print(f"\nEpoch {epoch + 1:03d}")
        print(f"Train Loss : {train_loss:.4f}")

        if test_loss is not None and test_mae is not None:
            print(f"Test Loss  : {test_loss:.4f}")
            print(f"Test MAE   : {test_mae:.4f}")
        else:
            print("Test Loss  : N/A")
            print("Test MAE   : N/A")

        print(f"LR         : {current_lr:.6f}")
        print(f"Time       : {duration:.1f}s")
        print(f"Curve Path : {curve_png_path}")

        is_best = False
        if test_mae is not None and test_mae < best_mae:
            best_mae = test_mae
            is_best = True

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_mae": test_mae,
            "best_test_mae": best_mae,
            "history": history
        }

        latest_path = os.path.join(Config.checkpoint_dir, Config.latest_model_name)
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(Config.checkpoint_dir, Config.best_model_name)
            torch.save(checkpoint, best_path)
            print("New best model saved")

    print("\nTraining complete")
    print(f"Training log saved to: {log_csv_path}")
    print(f"Training curve saved to: {curve_png_path}")


# ------------------------------
# Entry
# ------------------------------
if __name__ == "__main__":
    train()