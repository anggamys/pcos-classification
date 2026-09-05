import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class PCOSDataset(Dataset):
    def __init__(self, root_dir, transform=None, denoise=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.denoise = denoise
        self.samples = []

        for label, subdir in enumerate(["noninfected", "infected"]):
            subdir_path = self.root_dir / subdir
            for img_name in sorted(subdir_path.glob("*.jpg")):
                self.samples.append((str(img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.denoise:
            from .preprocessing import wavelet_denoise
            image = wavelet_denoise(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(train=True, image_size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


def create_dataloaders(root_dir, batch_size=32, image_size=224,
                       train_ratio=0.7, val_ratio=0.15, denoise=False,
                       num_workers=4):
    train_transform = get_transforms(train=True, image_size=image_size)
    val_transform = get_transforms(train=False, image_size=image_size)

    full_dataset = PCOSDataset(root_dir, transform=None, denoise=denoise)

    total = len(full_dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
