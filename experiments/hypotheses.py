"""Descriptive H1-H3 verdicts from experiment metrics (no statistics)."""

H3_TOLERANCE = 0.005


def _take(by_id: dict, row_id: str, keys: tuple) -> dict | None:
    """Ambil sekelompok metrik; None bila ada yang hilang.

    Args:
        by_id (dict): Metrik per id eksperimen.
        row_id (str): Id eksperimen (mis. "E4").
        keys (tuple): Kunci metrik yang diambil.

    Returns:
        dict atau None: Dict {key: float} bila lengkap, else None.
    """
    """Return {key: value} or None when any metric is missing."""
    values = {}
    for key in keys:
        value = by_id.get(row_id, {}).get(key)

        if value is None:
            return None

        values[key] = float(value)

    return values


def _optional(by_id: dict, row_id: str, key: str) -> float | None:
    """Ambil satu metrik opsional sebagai float.

    Args:
        by_id (dict): Metrik per id eksperimen.
        row_id (str): Id eksperimen.
        key (str): Kunci metrik.

    Returns:
        float atau None: Nilai metrik, atau None bila hilang.
    """
    """Return a single metric value or None when missing."""
    value = by_id.get(row_id, {}).get(key)
    return None if value is None else float(value)


def evaluate_hypotheses(rows: list) -> dict:
    """Verdict deskriptif H1-H3 dari metrik eksperimen (tanpa statistik).

    Hipotesis yang datanya belum lengkap dilewati agar run parsial
    (--only) tetap menghasilkan verdict untuk yang tersedia.

    Args:
        rows (list): Baris hasil tiap eksperimen (id + metrik).

    Returns:
        dict: Status accepted/not_accepted beserta detail per hipotesis.
    """
    by_id = {row["id"]: row for row in rows}
    verdicts = {}

    baseline = _take(by_id, "E1", ("accuracy", "auc"))
    denoised = _take(by_id, "E2", ("accuracy", "auc"))

    if baseline is not None and denoised is not None:
        accepted = (
            denoised["accuracy"] >= baseline["accuracy"]
            and denoised["auc"] >= baseline["auc"]
        )

        verdicts["H1"] = {
            "status": "accepted" if accepted else "not_accepted",
            "detail": "akurasi/AUC E2 >= E1 (+PSNR/SSIM dari quality.csv)",
        }

    full_image = _take(by_id, "E3", ("f1", "auc"))
    roi_runs = {
        experiment_id: _take(by_id, experiment_id, ("f1", "auc"))
        for experiment_id in ("E4", "E5")
    }

    if full_image is not None and all(roi_runs.values()):
        complete_roi_runs = {
            experiment_id: metrics
            for experiment_id, metrics in roi_runs.items()
            if metrics is not None
        }

        accepted = any(
            item["f1"] > full_image["f1"] or item["auc"] > full_image["auc"]
            for item in complete_roi_runs.values()
        )

        best_id = max(
            complete_roi_runs,
            key=lambda experiment_id: (
                complete_roi_runs[experiment_id]["f1"],
                complete_roi_runs[experiment_id]["auc"],
            ),
        )

        verdicts["H2"] = {
            "status": "accepted" if accepted else "not_accepted",
            "detail": f"F1/AUC {best_id} vs E3",
        }

    self_attention = _take(by_id, "E4", ("accuracy", "auc", "total_params"))
    cbam = _take(by_id, "E6", ("accuracy", "auc", "total_params"))
    self_attention_latency = _optional(by_id, "E4", "infer_ms_cpu")
    cbam_latency = _optional(by_id, "E6", "infer_ms_cpu")

    if self_attention is not None and cbam is not None:
        latency_ok = (
            self_attention_latency is None
            or cbam_latency is None
            or cbam_latency <= self_attention_latency
        )

        accepted = (
            cbam["accuracy"] >= self_attention["accuracy"] - H3_TOLERANCE
            and cbam["auc"] >= self_attention["auc"] - H3_TOLERANCE
            and cbam["total_params"] < self_attention["total_params"]
            and latency_ok
        )

        verdicts["H3"] = {
            "status": "accepted" if accepted else "not_accepted",
            "detail": "akurasi/AUC E6 >= E4 - 0.005 dengan parameter lebih sedikit",
        }

    return verdicts
