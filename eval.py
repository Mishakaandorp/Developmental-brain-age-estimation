import os
import csv
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from monai.transforms import Compose, ToTensor, ScaleIntensity
from monai.networks.nets import DenseNet201

# ------------------------------
# Config
# ------------------------------
class Config:
    test_dir = "/media/m-ssd5/Synthseg_BA_cs6/GitHub_BrainAge/Datasets/FETA_dHCP/T2_test"
    test_tsv = "/media/m-ssd5/Synthseg_BA_cs6/GitHub_BrainAge/Datasets/FETA_dHCP/test.tsv"

    model_name = "Model_A"
    # model_path = f"./Trained_models/{model_name}.pth"
    model_path = f"./Trained_models/Model_A/{model_name}.pth"

    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Utilities
# ------------------------------
def load_ages_from_tsv(tsv_path):
    ages = []
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            age = row.get('scan_age_mri0gamri') or row.get('scan_age')
            if age is None or age == "":
                continue
            ages.append(float(age))
    return np.array(ages)


def load_niftis(directory):
    files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])
    return files


# ------------------------------
# Dataset
# ------------------------------
class BrainAgeDataset(Dataset):
    def __init__(self, image_paths, ages):
        assert len(image_paths) == len(ages)

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
        img = nib.load(self.image_paths[idx]).get_fdata()
        age = self.ages[idx]

        img = self.transform(img)

        return img, torch.tensor(age, dtype=torch.float32)


# ------------------------------
# Model (MATCH TRAINING)
# ------------------------------
class AgePredictionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.densenet = DenseNet201(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            dropout_prob=0
        )

    def forward(self, x):
        return self.densenet(x).squeeze()


# ------------------------------
# Safe checkpoint loading
# ------------------------------
def load_model_weights(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model weights
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Clean DataParallel "module." prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k.replace("module.", "")] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()

    return model

def make_dir(model_name):
    save_dir = os.path.join("eval", model_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# ------------------------------
# Evaluation
# ------------------------------
def evaluate():

    print("Loading data...")

    test_paths = load_niftis(Config.test_dir)
    test_ages = load_ages_from_tsv(Config.test_tsv)

    print(f"Test samples: {len(test_paths)}")

    loader = DataLoader(
        BrainAgeDataset(test_paths, test_ages),
        batch_size=Config.batch_size,
        shuffle=False
    )

    print("Loading model...")

    model = AgePredictionModel().to(Config.device)
    model = load_model_weights(model, Config.model_path, Config.device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(Config.device)
            out = model(x).view(-1)

            preds.append(out.cpu())
            targets.append(y)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    # ------------------------------
    # Metrics
    # ------------------------------
    errors = preds - targets
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    bag_mean = errors.mean()
    bag_std = errors.std()

    print("\nEvaluation results:")
    print(f"MAE      : {mae:.4f}")
    print(f"BAG mean : {bag_mean:.4f}")
    print(f"BAG std  : {bag_std:.4f}")

    # ------------------------------
    # Output directory (FIXED NAME)
    # ------------------------------
    save_dir = make_dir(Config.model_name)
    

    # ------------------------------
    # Plot: BAG violin plot
    # ------------------------------
    plt.figure()
    plt.violinplot(errors)

    plt.title(f"BAG Distribution | MAE={mae:.3f}, BAG mean={bag_mean:.3f}")
    plt.ylabel("BAG (Predicted - True Age)")

    bag_plot_path = os.path.join(save_dir, "bag_violin.png")
    plt.savefig(bag_plot_path)
    plt.close()

    print("\nSaved plots:")
    print(bag_plot_path)


# ------------------------------
# Entry
# ------------------------------
if __name__ == "__main__":
    evaluate()
