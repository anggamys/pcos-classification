"""Pemilihan device komputasi: GPU diutamakan, CPU sebagai fallback.

Default ``auto`` memakai CUDA bila tersedia sehingga run di Colab langsung
menggunakan GPU T4. Di mesin lokal dengan torch CPU-only, ``auto`` turun ke
CPU tanpa mengubah kode. ``--device cuda`` bersifat ketat: bila CUDA tidak
tersedia, run berhenti dengan pesan yang jelas alih-alih diam-diam memakai CPU.
"""

import torch

DEVICE_CHOICES = ("auto", "cuda", "cpu")


def resolve_device(preference: str = "auto") -> torch.device:
    """Pilih device komputasi dengan GPU sebagai prioritas default.

    Args:
        preference (str): ``"auto"`` (CUDA bila tersedia, selain itu CPU),
            ``"cuda"`` (wajib GPU), atau ``"cpu"`` (paksa CPU).

    Returns:
        torch.device: Device yang dipakai training dan evaluasi.

    Raises:
        ValueError: Jika nilai preferensi tidak dikenal.
        RuntimeError: Jika ``preference="cuda"`` tetapi CUDA tidak tersedia,
            misalnya karena torch yang terpasang adalah build CPU-only.
    """
    if preference not in DEVICE_CHOICES:
        raise ValueError(
            f"Unknown device preference: {preference}. "
            f"Pilih salah satu dari {', '.join(DEVICE_CHOICES)}"
        )

    if preference == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if preference == "cuda":
        raise RuntimeError(
            "CUDA diminta tetapi tidak tersedia. Pasang torch build CUDA "
            "(lihat https://pytorch.org/get-started/locally/) atau jalankan "
            "dengan --device cpu."
        )

    return torch.device("cpu")
