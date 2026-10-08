from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .preprocessing.quality import enhance
from .preprocessing.segmentation import apply_masked, apply_roi_crop, segment
from .preprocessing.wavelet import wavelet_denoise


class PCOSDataset(Dataset):
    """Dataset citra USG ovarium (PCOS vs sehat) + preprocessing PCD.

    Args:
        root_dir (str atau Path): Direktori berisi subfolder `infected`
            (label 1) dan `noninfected` (label 0).
        transform: Transform torchvision yang diterapkan setelah preprocessing.
        denoise (bool): Terapkan wavelet denoising BayesShrink.
        enhance (str): "none" atau "clahe".
        segment (str): "none", "otsu", atau "adaptive".
        input_mode (str): "full" (citra utuh), "roi" (crop), atau "masked".
        seg_pad (int): Padding ROI dalam piksel.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform=None,
        denoise: bool = False,
        enhance: str = "none",
        segment: str = "none",
        input_mode: str = "full",
        seg_pad: int = 8,
    ):
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

    def __len__(self) -> int:
        """Cacah seluruh sampel (infected + noninfected)."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """Ambil satu sampel: preprocessing PCD lalu transform.

        Args:
            idx (int): Indeks sampel.

        Returns:
            tuple: Pasangan (citra tensor/PIL, label 0/1).
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.enhance != "none":
            image = enhance(image, method=self.enhance)

        if self.denoise:
            image = wavelet_denoise(image)

        if self.input_mode in ("roi", "masked") or self.segment != "none":
            mask = segment(image, method=self.segment)
            if self.input_mode == "roi":
                image = apply_roi_crop(image, mask, pad=self.seg_pad)
            elif self.input_mode == "masked":
                image = apply_masked(image, mask)

        if self.transform:
            image = self.transform(image)

        return image, label


class TransformedSubset(Dataset):
    """Subset indeks dengan transform terpisah untuk train/val/test.

    Args:
        dataset (Dataset): Dataset sumber yang sudah di-preprocessing.
        indices: Daftar indeks sampel subset ini.
        transform: Transform torchvision khusus subset ini.
    """

    def __init__(self, dataset: Dataset, indices, transform) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        """Cacah sampel dalam subset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        """Ambil sampel subset lalu terapkan transform.

        Args:
            idx (int): Indeks dalam subset (bukan indeks dataset).

        Returns:
            tuple: Pasangan (citra, label).
        """
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train: bool = True, image_size: int = 224) -> transforms.Compose:
    """Susun transform torchvision untuk train atau evaluasi.

    Train memakai augmentasi (rotasi, flip, crop acak); evaluasi hanya
    resize + normalisasi ImageNet.

    Args:
        train (bool): True untuk pipeline training beraugmentasi.
        image_size (int): Ukuran resize persegi (default 224).

    Returns:
        torchvision.transforms.Compose: Rangkaian transform siap pakai.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


def create_dataloaders(
    root_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    denoise: bool = False,
    enhance: str = "none",
    segment: str = "none",
    input_mode: str = "full",
    seg_pad: int = 8,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Bagi dataset (seed 42) dan bangun tiga DataLoader train/val/test.

    Args:
        root_dir (str atau Path): Direktori dataset.
        batch_size (int): Ukuran batch.
        image_size (int): Ukuran resize persegi.
        train_ratio (float): Proporsi train.
        val_ratio (float): Proporsi validasi (sisanya untuk test).
        denoise (bool): Wavelet denoising BayesShrink.
        enhance (str): "none" atau "clahe".
        segment (str): "none", "otsu", atau "adaptive".
        input_mode (str): "full", "roi", atau "masked".
        seg_pad (int): Padding ROI dalam piksel.
        num_workers (int): Pekerja DataLoader.

    Returns:
        tuple: (train_loader, val_loader, test_loader).
    """
    train_transform = get_transforms(train=True, image_size=image_size)
    val_transform = get_transforms(train=False, image_size=image_size)

    full_dataset = PCOSDataset(
        root_dir,
        transform=None,
        denoise=denoise,
        enhance=enhance,
        segment=segment,
        input_mode=input_mode,
        seg_pad=seg_pad,
    )

    total = len(full_dataset)
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(42))

    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_ds = TransformedSubset(full_dataset, train_indices, train_transform)
    val_ds = TransformedSubset(full_dataset, val_indices, val_transform)
    test_ds = TransformedSubset(full_dataset, test_indices, val_transform)

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader
