from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


class PCOSDataset(Dataset):
    def __init__(self, root_dir, transform=None, denoise=False,
                 enhance="none", segment="none", input_mode="full",
                 seg_pad=8):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.denoise = denoise
        self.enhance = enhance
        self.segment = segment
        self.input_mode = input_mode
        self.seg_pad = seg_pad
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

        if self.enhance != "none":
            from .image_quality import enhance
            image = enhance(image, method=self.enhance)

        if self.denoise:
            from .preprocessing import wavelet_denoise
            image = wavelet_denoise(image)

        if self.input_mode in ("roi", "masked") or self.segment != "none":
            from .segmentation import segment, apply_roi_crop, apply_masked
            mask = segment(image, method=self.segment)
            if self.input_mode == "roi":
                image = apply_roi_crop(image, mask, pad=self.seg_pad)
            elif self.input_mode == "masked":
                image = apply_masked(image, mask)

        if self.transform:
            image = self.transform(image)

        return image, label


class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


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
                       enhance="none", segment="none", input_mode="full",
                       seg_pad=8, num_workers=4):
    train_transform = get_transforms(train=True, image_size=image_size)
    val_transform = get_transforms(train=False, image_size=image_size)

    full_dataset = PCOSDataset(root_dir, transform=None, denoise=denoise,
                             enhance=enhance, segment=segment,
                             input_mode=input_mode, seg_pad=seg_pad)

    total = len(full_dataset)
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(42))

    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_ds = TransformedSubset(full_dataset, train_indices, train_transform)
    val_ds = TransformedSubset(full_dataset, val_indices, val_transform)
    test_ds = TransformedSubset(full_dataset, test_indices, val_transform)

    persistent = num_workers > 0

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=persistent)

    return train_loader, val_loader, test_loader
