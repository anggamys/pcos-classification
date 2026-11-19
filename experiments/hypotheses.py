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

Untuk beberapa seed, kriteria di atas tidak diubah. Metrik agregat adalah
rata-rata aritmetika antar seed, verdict dihitung pada baris rata-rata, dan
status akhir accepted menuntut dukungan mayoritas seed (lihat
`evaluate_replicated_hypotheses`). Aturan ini ditetapkan sebelum run
multi-seed, bukan setelah melihat hasilnya.
"""

import statistics

METRIC_TOLERANCE = 0.005

# Metrik yang dirata-ratakan lintas seed. `total_params` sengaja tidak ikut
# karena properti arsitektur, bukan hasil run; latency diukur sekali per
# matriks dan dipakai bersama semua seed.
AGGREGATE_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "auc",
    "infer_ms_cpu",
)


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

        if cbam_latency is None or self_attention_latency is None:
            latency_detail = "latency tidak dievaluasi (data tidak tersedia)"
        else:
            latency_detail = (
                f"latency {cbam_latency:.2f} ms vs {self_attention_latency:.2f} ms "
                "(syarat: tidak lebih besar)"
            )

        verdicts["H3"] = {
            "status": "accepted" if accepted else "not_accepted",
            "detail": (
                "E6 vs E4: akurasi/AUC >= E4 - 0.005, "
                f"params {int(cbam['total_params']):,} vs "
                f"{int(self_attention['total_params']):,}, {latency_detail}"
            ),
        }

    return verdicts


def aggregate_seed_rows(per_seed_rows: dict) -> tuple[list, list]:
    """Rata-ratakan metrik tiap eksperimen lintas seed.

    Metrik non-numerik (id, desc, attention) disalin dari seed pertama.
    Simpangan baku memakai rumus sampel (n-1); dengan satu seed nilainya 0.

    Args:
        per_seed_rows (dict): Pemetaan label seed (mis. "42") ke daftar baris
            hasil berformat `results.csv`.

    Returns:
        tuple: (mean_rows, sd_rows). `mean_rows` dipakai untuk verdict,
            `sd_rows` memuat simpangan baku per metrik.
    """
    labels = list(per_seed_rows)

    if not labels:
        return [], []

    by_seed = {
        label: {row["id"]: row for row in rows}
        for label, rows in per_seed_rows.items()
    }
    mean_rows = []
    sd_rows = []

    for experiment_id in [row["id"] for row in per_seed_rows[labels[0]]]:
        template = by_seed[labels[0]][experiment_id]
        mean_row = dict(template)
        sd_row = {"id": experiment_id, "desc": template.get("desc", "")}

        for metric in AGGREGATE_METRICS:
            values = [
                float(by_seed[label][experiment_id][metric])
                for label in labels
                if metric in by_seed[label].get(experiment_id, {})
                and by_seed[label][experiment_id][metric] is not None
            ]

            if not values:
                continue

            mean_row[metric] = sum(values) / len(values)
            sd_row[metric] = statistics.stdev(values) if len(values) > 1 else 0.0

        mean_rows.append(mean_row)
        sd_rows.append(sd_row)

    return mean_rows, sd_rows


def evaluate_replicated_hypotheses(per_seed_rows: dict) -> dict:
    """Verdict H1-H3 pada agregat seed dengan syarat dukungan mayoritas.

    Aturan prespecified:
    - Metrik agregat = rata-rata aritmetika antar seed.
    - Verdict agregat memakai kriteria H1-H3 yang sama seperti
      `evaluate_hypotheses`, tanpa mengubah toleransi atau varian utama.
    - Status akhir accepted hanya bila verdict agregat accepted dan minimal
      ``n // 2 + 1`` seed juga accepted. Aturan ini mencegah satu run yang
      kebetulan lolos menyelamatkan klaim.
    - E5 tetap hanya dilaporkan deskriptif.

    Args:
        per_seed_rows (dict): Pemetaan label seed ke daftar baris hasil.

    Returns:
        dict: Verdict akhir, verdict agregat, verdict per seed, serta baris
            rata-rata dan simpangan baku.
    """
    labels = list(per_seed_rows)
    mean_rows, sd_rows = aggregate_seed_rows(per_seed_rows)
    aggregate = evaluate_hypotheses(mean_rows)
    per_seed = {
        label: evaluate_hypotheses(rows) for label, rows in per_seed_rows.items()
    }
    required = len(labels) // 2 + 1
    hypotheses = {}

    for name, verdict in aggregate.items():
        support = sum(
            1
            for seed_verdicts in per_seed.values()
            if seed_verdicts.get(name, {}).get("status") == "accepted"
        )
        accepted = verdict["status"] == "accepted" and support >= required
        hypotheses[name] = {
            "status": "accepted" if accepted else "not_accepted",
            "detail": (
                f"{verdict['detail']}; dukungan seed {support}/{len(labels)} "
                f"(syarat {required}/{len(labels)})"
            ),
            "support": f"{support}/{len(labels)}",
        }

    return {
        "n_seeds": len(labels),
        "seeds": labels,
        "required_support": required,
        "hypotheses": hypotheses,
        "aggregate_hypotheses": aggregate,
        "per_seed_hypotheses": per_seed,
        "mean_rows": mean_rows,
        "sd_rows": sd_rows,
    }
