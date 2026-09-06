import torch
import numpy as np
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset, DataLoader


class SMOTEDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def extract_features(model, dataloader, device, max_samples=1000):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i * images.size(0) >= max_samples:
                break

            images = images.to(device)

            if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
                feat = model.backbone.features(images)
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
                feat = feat.view(feat.size(0), -1)
            else:
                feat = images.view(images.size(0), -1)

            features_list.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels


def apply_smote_to_features(features, labels, random_state=42):
    n_samples, n_features = features.shape

    k_neighbors = min(5, n_samples - 1)

    smote = SMOTE(
        sampling_strategy="auto",
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    print(f"Original: {len(labels)} samples")
    print(f"Resampled: {len(labels_resampled)} samples")
    print(f"Class distribution: {np.bincount(labels_resampled.astype(int))}")

    return features_resampled, labels_resampled


def create_smote_dataloader(model, train_loader, device, batch_size=32,
                            image_size=224, max_samples=2000):
    print("Extracting features for SMOTE...")
    features, labels = extract_features(model, train_loader, device, max_samples)

    print("Applying SMOTE...")
    features_resampled, labels_resampled = apply_smote_to_features(features, labels)

    features_tensor = torch.tensor(features_resampled, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_resampled, dtype=torch.long)

    smote_dataset = SMOTEDataset(features_tensor, labels_tensor)
    smote_loader = DataLoader(smote_dataset, batch_size=batch_size, shuffle=True)

    return smote_loader
