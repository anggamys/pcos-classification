import hashlib
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
        dedupe (bool): Buang salinan byte-identik (hash md5, keep-first)
            sebelum split agar test set tidak terkontaminasi salinan train.

    Catatan: hasil preprocessing deterministik per citra di-cache in-memory
    (terisi saat epoch pertama; dengan persistent_workers cache bertahan
    sepanjang training). Transform augmentasi tidak ikut ter-cache.
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
        dedupe: bool = True,
    ):
        self.root_dir = Path(root_dir)

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Direktori dataset tidak ditemukan: {self.root_dir}"
            )

        self.transform = transform
        self.denoise = denoise
        self.enhance = enhance
        self.segment = segment
        self.input_mode = input_mode
        self.seg_pad = seg_pad
        self.dedup_stats: dict | None = None
        self._cache: dict = {}
        self.samples = []

        for label, subdir in enumerate(["noninfected", "infected"]):
            subdir_path = self.root_dir / subdir
            matches = sorted(subdir_path.glob("*.jpg"))

            if not matches:
                raise FileNotFoundError(
                    f"Tidak ada citra .jpg di {subdir_path} (label '{subdir}'). "
                    "Struktur yang diharapkan: <root>/noninfected dan "
                    "<root>/infected berisi file .jpg."
                )

            for img_name in matches:
                self.samples.append((str(img_name), label))

        if dedupe:
            self._remove_duplicates()

        # Dedup keep-first bisa memusnahkan seluruh kelas bila semua
        # citranya salinan identik dari kelas lain; jangan biarkan senyap.
        labels_present = {label for _, label in self.samples}

        if len(labels_present) < 2:
            raise ValueError(
                "Dataset hanya memiliki "
                f"{sorted(labels_present)} kelas (butuh noninfected dan "
                "infected). Cek direktori dataset dan duplikasi antar-kelas."
            )

    def _remove_duplicates(self) -> None:
        """Buang salinan byte-identik dari daftar sampel (md5, keep-first).

        Statistik dedup disimpan ke `self.dedup_stats` untuk ketertelusuran
        run (dicatat ulang di run_summary.json).
        """
        seen: set[str] = set()
        unique = []

        for path_str, label in self.samples:
            digest = hashlib.md5(Path(path_str).read_bytes()).hexdigest()

            if digest not in seen:
                seen.add(digest)
                unique.append((path_str, label))

        raw = len(self.samples)
        self.samples = unique
        self.dedup_stats = {
            "raw": raw,
            "unique": len(unique),
            "removed": raw - len(unique),
        }
        print(
            f"Dedup md5: {raw} citra mentah -> {len(unique)} unik "
            f"({raw - len(unique)} salinan identik dibuang)"
        )

    def __len__(self) -> int:
        """Cacah seluruh sampel (infected + noninfected)."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """Ambil satu sampel: preprocessing PCD lalu transform.

        Preprocessing deterministik (denoise/enhance/segmentasi) di-cache
        in-memory per citra agar tidak dihitung ulang tiap epoch. Transform
        augmentasi tetap diterapkan setiap panggilan sehingga augmentasi
        acak tidak ikut ter-cache.

        Args:
            idx (int): Indeks sampel.

        Returns:
            tuple: Pasangan (citra tensor/PIL, label 0/1).
        """
        img_path, label = self.samples[idx]

        if img_path in self._cache:
            image, mask = self._cache[img_path]
        else:
            image = Image.open(img_path).convert("RGB")

            if self.enhance != "none":
                image = enhance(image, method=self.enhance)

            if self.denoise:
                image = wavelet_denoise(image)

            if self.input_mode in ("roi", "masked") or self.segment != "none":
                mask = segment(image, method=self.segment)
            else:
                mask = None

            if self.denoise or self.enhance != "none" or mask is not None:
                self._cache[img_path] = (image, mask)

        if self.input_mode == "roi" and mask is not None:
            image = apply_roi_crop(image, mask, pad=self.seg_pad)
        elif self.input_mode == "masked" and mask is not None:
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
    dedupe: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Bagi dataset (seed tetap) dan bangun tiga DataLoader train/val/test.

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
        dedupe (bool): Buang salinan byte-identik sebelum split.
        seed (int): Seed split dan shuffle (default 42). Satu seed dipakai
            untuk seluruh eksperimen agar perbandingan memakai sampel sama.

    Returns:
        tuple: (train_loader, val_loader, test_loader).

    Raises:
        ValueError: Jika rasio split tidak valid (train_ratio + val_ratio
            harus < 1) atau dataset terlalu kecil sehingga ada subset kosong.
    """
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError(
            f"Rasio split tidak valid: train_ratio={train_ratio}, "
            f"val_ratio={val_ratio} (butuh 0 < train_ratio < 1, "
            "0 <= val_ratio < 1)"
        )

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            f"train_ratio + val_ratio harus < 1 agar test set tidak kosong "
            f"(diterima {train_ratio} + {val_ratio})"
        )

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
        dedupe=dedupe,
    )

    total = len(full_dataset)
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed))

    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)

    if min(train_size, val_size, total - train_size - val_size) < 1:
        raise ValueError(
            f"Dataset terlalu kecil untuk split ini: total={total} -> "
            f"train={train_size}, val={val_size}, "
            f"test={total - train_size - val_size}. "
            "Kurangi rasio atau tambahkan data."
        )

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
        generator=torch.Generator().manual_seed(seed),
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
