"""Descriptive H1-H3 verdicts from experiment metrics (no statistics).

Operasionalisasi yang disepakati (prespecified, bukan post-hoc):
- H1: E2 harus lebih baik secara ketat pada minimal satu metrik
  (akurasi atau AUC) tanpa regresi melebihi toleransi pada metrik
  satunya. Hasil identik tidak dihitung accepted. PSNR/SSIM dari
  quality.csv berperan sebagai bukti deskriptif, bukan penentu verdict.
- H2: varian utama yang diprespesifikasi adalah E4 (ROI crop Otsu)
  dengan AUC sebagai metrik utama; E5 (masked) dilaporkan deskriptif
  agar tidak terjadi seleksi varian terbaik di atas test set.
- H3: E6 vs E4 dengan toleransi 0.005 pada akurasi/AUC.
"""

METRIC_TOLERANCE = 0.005


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

    Kriteria prespecified:
    - H1: E2 > E1 secara ketat pada minimal satu metrik (akurasi atau AUC)
      tanpa regresi melebihi toleransi pada metrik satunya.
    - H2: AUC E4 > AUC E3 dan F1 E4 tidak lebih buruk melebihi toleransi;
      E5 hanya pelaporan deskriptif (bukan bagian verdict).
    - H3: E6 >= E4 - toleransi pada akurasi/AUC, parameter lebih sedikit,
      latency tidak lebih besar.

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
        acc_improves = (
            denoised["accuracy"] > baseline["accuracy"]
            and denoised["auc"] >= baseline["auc"] - METRIC_TOLERANCE
        )
        auc_improves = (
            denoised["auc"] > baseline["auc"]
            and denoised["accuracy"] >= baseline["accuracy"] - METRIC_TOLERANCE
        )

        verdicts["H1"] = {
            "status": "accepted" if (acc_improves or auc_improves) else "not_accepted",
            "detail": (
                "E2 > E1 ketat pada minimal satu metrik (akurasi/AUC) tanpa "
                "regresi > 0.005; PSNR/SSIM dari quality.csv deskriptif"
            ),
        }

    full_image = _take(by_id, "E3", ("f1", "auc"))
    roi_primary = _take(by_id, "E4", ("f1", "auc"))
    masked = _take(by_id, "E5", ("f1", "auc"))

    if full_image is not None and roi_primary is not None:
        accepted = (
            roi_primary["auc"] > full_image["auc"]
            and roi_primary["f1"] >= full_image["f1"] - METRIC_TOLERANCE
        )

        detail = (
            "AUC E4 > E3 dan F1 E4 >= E3 - 0.005 (varian utama prespecified); "
            "E5 masked deskriptif"
        )

        if masked is not None:
            detail += f": F1 {masked['f1']:.4f}, AUC {masked['auc']:.4f}"

        verdicts["H2"] = {
            "status": "accepted" if accepted else "not_accepted",
            "detail": detail,
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
            cbam["accuracy"] >= self_attention["accuracy"] - METRIC_TOLERANCE
            and cbam["auc"] >= self_attention["auc"] - METRIC_TOLERANCE
            and cbam["total_params"] < self_attention["total_params"]
            and latency_ok
        )

        verdicts["H3"] = {
            "status": "accepted" if accepted else "not_accepted",
            "detail": "akurasi/AUC E6 >= E4 - 0.005 dengan parameter lebih sedikit",
        }

    return verdicts
