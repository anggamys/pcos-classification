"""Experiment matrix configuration: YAML loading and validation."""

from pathlib import Path

import yaml

VALID_CHOICES = {
    "enhance": ["none", "clahe"],
    "segment": ["none", "otsu", "adaptive"],
    "input_mode": ["full", "roi", "masked"],
    "attention": ["self_attention", "se_net", "cbam", "transformer"],
}

REQUIRED_KEYS = ["id", "desc", "denoise"] + list(VALID_CHOICES)


def load_config(config_path):
    """Muat dan validasi YAML matriks eksperimen sebagai sumber tunggal.

    Args:
        config_path (str atau Path): Path file experiments.yml.

    Returns:
        tuple: (defaults, experiments) dengan tiap eksperimen sudah
            digabung bersama defaults.

    Raises:
        ValueError: Bila ada key wajib hilang, pilihan tak valid,
            atau id eksperimen ganda.
    """
    with open(config_path) as handle:
        raw = yaml.safe_load(handle)
    defaults = raw.get("defaults", {})
    experiments = []
    for position, item in enumerate(raw.get("experiments", []), start=1):
        merged = dict(defaults)
        merged.update(item)
        missing = [key for key in REQUIRED_KEYS if key not in merged]
        if missing:
            raise ValueError(f"Experiment #{position} missing keys: {missing}")
        for key, choices in VALID_CHOICES.items():
            if merged[key] not in choices:
                raise ValueError(
                    f"Experiment {merged.get('id', position)}: invalid {key} "
                    f"{merged[key]!r}, expected one of {choices}"
                )
        experiments.append(merged)
    ids = [experiment["id"] for experiment in experiments]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate experiment ids: {ids}")
    return defaults, experiments


def resolve_overrides(args, experiments):
    """Terapkan flag CLI eksplisit di atas nilai YAML.

    Args:
        args (argparse.Namespace): Argumen CLI (hanya yang tak-None dipakai).
        experiments (list): Daftar dict eksperimen hasil `load_config`.

    Returns:
        list: Daftar eksperimen yang sama, sudah ditimpa override.
    """
    """Apply explicit CLI flags on top of the YAML values."""
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in ("data_dir", "epochs", "batch_size", "gradcam_samples", "out_dir")
        and value is not None
    }
    for experiment in experiments:
        experiment.update(overrides)
    return experiments


def select_experiments(experiments, only):
    """Saring eksperimen berdasarkan flag --only.

    Args:
        experiments (list): Daftar dict eksperimen.
        only (str): Daftar id koma-pisah (mis. "E1,E2"); kosong = semua.

    Returns:
        list: Subset eksperimen terpilih.

    Raises:
        ValueError: Bila ada id tak dikenal, disertai id yang valid.
    """
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    known_ids = [experiment["id"] for experiment in experiments]
    unknown = wanted - set(known_ids)
    if unknown:
        raise ValueError(
            f"Unknown experiment ids: {sorted(unknown)}, valid: {known_ids}"
        )
    selected = [
        experiment
        for experiment in experiments
        if not wanted or experiment["id"] in wanted
    ]
    return selected


def project_root():
    """Path akar proyek (induk folder experiments).

    Returns:
        pathlib.Path: Direktori akar proyek.
    """
    return Path(__file__).resolve().parent.parent
