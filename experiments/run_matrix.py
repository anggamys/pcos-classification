"""Run the 6-experiment matrix for the dual-course final project.

Configuration lives in a YAML file (single source of truth):

    python experiments/run_matrix.py --config experiments/experiments.yml
    python experiments/run_matrix.py --config experiments/experiments.yml --only E1,E2

Multi-seed dengan agregasi mean +/- SD (layout `seed-<n>/E*`):

    python experiments/run_matrix.py --seeds 42,43,44

CLI flags (--data-dir, --epochs, --batch-size, --out-dir, --gradcam-samples)
override the YAML defaults when explicitly passed.
"""

import argparse
import codecs
import csv
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from io import BufferedReader
from pathlib import Path
from typing import cast

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from exp_config import (
    load_config,
    project_root,
    resolve_overrides,
    select_experiments,
)
from hypotheses import (
    AGGREGATE_METRICS,
    evaluate_hypotheses,
    evaluate_replicated_hypotheses,
)

from src.models.densenet121 import DenseNet121Attention

PROJECT_ROOT = project_root()
SEED_DIR_PREFIX = "seed-"
LATENCY_FILE = "latency.json"
NUMERIC_COLUMNS = {
    "returncode",
    "seed",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "auc",
    "total_params",
    "infer_ms_cpu",
}
INTEGER_COLUMNS = {"returncode", "seed", "total_params"}

MAIN_FLAGS = [
    ("--data-dir", "data_dir"),
    ("--epochs", "epochs"),
    ("--batch-size", "batch_size"),
    ("--image-size", "image_size"),
    ("--lr", "lr"),
    ("--weight-decay", "weight_decay"),
    ("--patience", "patience"),
    ("--dropout", "dropout"),
    ("--seg-pad", "seg_pad"),
    ("--num-workers", "num_workers"),
    ("--seed", "seed"),
    ("--attention", "attention"),
    ("--enhance", "enhance"),
    ("--segment", "segment"),
    ("--input-mode", "input_mode"),
    ("--gradcam-samples", "gradcam_samples"),
    ("--device", "device"),
]

METRIC_RE = {
    "accuracy": re.compile(r"Accuracy:\s+([0-9.]+)"),
    "precision": re.compile(r"Precision:\s+([0-9.]+)"),
    "recall": re.compile(r"Recall:\s+([0-9.]+)"),
    "f1": re.compile(r"F1-Score:\s+([0-9.]+)"),
    "specificity": re.compile(r"Specificity:\s+([0-9.]+)"),
    "auc": re.compile(r"AUC:\s+([0-9.]+)"),
}


def parse_metrics(stdout: str) -> dict:
    """Urai metrik (Accuracy/AUC/...) dari stdout main.py via regex.

    Args:
        stdout (str): Keluaran terminal satu run main.py.

    Returns:
        dict: Nilai float per metrik; None bila pola tak ditemukan.
    """
    metrics = {}
    for key, pattern in METRIC_RE.items():
        match = pattern.search(stdout)
        metrics[key] = float(match.group(1)) if match else None
    return metrics


def model_stats(attention: str, repeats: int = 20) -> tuple[int, float]:
    """Hitung parameter dan latency inferensi CPU (bukti efisiensi H3).

    Args:
        attention (str): Jenis mekanisme attention model.
        repeats (int): Pengulangan pengukuran latency (default 20).

    Returns:
        tuple: (total parameter, latency ms per forward di CPU).
    """
    model = DenseNet121Attention(pretrained=False, attention_type=attention)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    sample_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_input)

        start_time = time.perf_counter()
        for _ in range(repeats):
            _ = model(sample_input)

        elapsed_ms = (time.perf_counter() - start_time) / repeats * 1000.0

    return total_params, round(elapsed_ms, 2)


def parse_seed_list(text: str) -> list:
    """Urai daftar seed koma-pisah menjadi bilangan bulat unik.

    Args:
        text (str): Daftar seed, mis. "42,43,44".

    Returns:
        list: Seed unik sesuai urutan masukan.

    Raises:
        ValueError: Bila kosong, bukan bilangan bulat, negatif, atau duplikat.
    """
    seeds = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        try:
            seed = int(item)
        except ValueError:
            raise ValueError(f"Seed harus bilangan bulat: {item!r}") from None

        if seed < 0:
            raise ValueError(f"Seed tidak boleh negatif: {seed}")

        seeds.append(seed)

    if not seeds:
        raise ValueError("Daftar seed kosong; contoh: --seeds 42,43,44")

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seed duplikat: {seeds}")

    return seeds


def _mean_forward_ms(model: torch.nn.Module, sample_input, repeats: int) -> float:
    """Waktu rata-rata satu forward, dalam milidetik.

    Args:
        model (torch.nn.Module): Model dalam mode eval.
        sample_input (torch.Tensor): Batch satu citra (1, 3, H, W).
        repeats (int): Cacah forward yang diukur.

    Returns:
        float: Rata-rata waktu per forward (ms).
    """
    with torch.inference_mode():
        start_time = time.perf_counter()

        for _ in range(repeats):
            _ = model(sample_input)

        return (time.perf_counter() - start_time) / repeats * 1000.0


def benchmark_latency(
    attention_types: list,
    rounds: int = 5,
    repeats: int = 20,
    model_factory=None,
) -> dict:
    """Ukur latency CPU beberapa attention secara bergantian.

    Setiap model dibangun sekali dan dipanaskan, lalu diukur bergantian per
    putaran. Pengukuran bergantian mencegah satu model menanggung beban CPU,
    cache, atau kondisi thermal yang berbeda dari model pembandingnya. Nilai
    yang dilaporkan adalah median dari `rounds` putaran.

    Args:
        attention_types (list): Jenis attention yang dibandingkan.
        rounds (int): Cacah putaran pengukuran (minimal 1).
        repeats (int): Cacah forward per putaran (minimal 1).
        model_factory (callable): Pembuat model; default DenseNet121Attention.
            Diganti model kecil saat pengujian unit.

    Returns:
        dict: Statistik per attention (median, min, max, samples, params).

    Raises:
        ValueError: Bila `rounds` atau `repeats` kurang dari 1.
    """
    if rounds < 1 or repeats < 1:
        raise ValueError("rounds dan repeats harus >= 1")

    factory = model_factory or DenseNet121Attention
    order = list(dict.fromkeys(attention_types))
    models = {}
    stats = {}

    for attention in order:
        model = factory(pretrained=False, attention_type=attention)
        model.eval()
        models[attention] = model
        stats[attention] = {
            "total_params": sum(p.numel() for p in model.parameters()),
            "samples_ms": [],
        }

    sample_input = torch.randn(1, 3, 224, 224)

    with torch.inference_mode():
        for model in models.values():
            for _ in range(3):
                _ = model(sample_input)

    for _ in range(rounds):
        for attention in order:
            stats[attention]["samples_ms"].append(
                _mean_forward_ms(models[attention], sample_input, repeats)
            )

    for attention in order:
        samples = stats[attention]["samples_ms"]
        stats[attention]["median_ms"] = round(statistics.median(samples), 2)
        stats[attention]["min_ms"] = round(min(samples), 2)
        stats[attention]["max_ms"] = round(max(samples), 2)
        stats[attention]["samples_ms"] = [round(value, 2) for value in samples]

    return stats


def load_or_measure_latency(
    out_root: Path,
    attention_types: list,
    rounds: int = 5,
    repeats: int = 20,
    remeasure: bool = False,
) -> dict:
    """Pakai latency.json yang ada atau ukur baru sekali per matriks.

    Satu pengukuran dipakai bersama semua seed agar verdict H3 tidak
    membandingkan angka dari sesi pengukuran yang berbeda.

    Args:
        out_root (Path): Direktori akar hasil matriks.
        attention_types (list): Jenis attention yang dibutuhkan.
        rounds (int): Cacah putaran pengukuran.
        repeats (int): Cacah forward per putaran.
        remeasure (bool): Paksa pengukuran ulang walau cache lengkap.

    Returns:
        dict: Payload latency.json (per_attention + metadata).
    """
    path = out_root / LATENCY_FILE
    cached = json.loads(path.read_text()) if path.exists() else None
    cached_attention = (cached or {}).get("per_attention", {})
    complete = all(item in cached_attention for item in attention_types)

    if cached is not None and complete and not remeasure:
        return cached

    payload = {
        "measured_at": datetime.now(UTC).isoformat(),
        "interleaved_rounds": rounds,
        "repeats_per_round": repeats,
        "torch_num_threads": torch.get_num_threads(),
        "per_attention": benchmark_latency(attention_types, rounds, repeats),
    }
    path.write_text(json.dumps(payload, indent=2))

    print(f"Saved {path}")

    return payload


def apply_latency(rows: list, latency: dict) -> list:
    """Timpa latency per run dengan hasil benchmark bersama.

    Args:
        rows (list): Baris hasil eksperimen (dimodifikasi di tempat).
        latency (dict): Payload dari `load_or_measure_latency`.

    Returns:
        list: Baris yang sama setelah latency dan parameter diperbarui.
    """
    per_attention = latency.get("per_attention", {})

    for row in rows:
        stats = per_attention.get(row.get("attention"))

        if stats is None:
            continue

        row["total_params"] = int(stats["total_params"])
        row["infer_ms_cpu"] = stats["median_ms"]

    return rows


def write_results_csv(path: Path, rows: list) -> None:
    """Tulis CSV hasil dengan kolom mengikuti kunci baris pertama.

    Args:
        path (Path): Berkas tujuan.
        rows (list): Baris hasil yang sudah lengkap.
    """
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path}")


def read_results_csv(path: Path) -> list:
    """Baca results.csv dan kembalikan angka sebagai float/int.

    Args:
        path (Path): Berkas results.csv.

    Returns:
        list: Baris hasil dengan kolom numerik sudah dikonversi.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for key in NUMERIC_COLUMNS:
            if key not in row:
                continue

            value = row[key]

            if value in (None, ""):
                row[key] = None
                continue

            number = float(value)
            row[key] = int(number) if key in INTEGER_COLUMNS else number

    return rows


def collect_seed_results(out_root: Path) -> dict:
    """Baca seluruh `seed-*/results.csv` di bawah direktori hasil.

    Args:
        out_root (Path): Direktori akar hasil matriks.

    Returns:
        dict: Pemetaan label seed ke daftar baris hasil, urut label.
    """
    per_seed = {}

    for csv_path in sorted(out_root.glob(f"{SEED_DIR_PREFIX}*/results.csv")):
        label = csv_path.parent.name.removeprefix(SEED_DIR_PREFIX)
        per_seed[label] = read_results_csv(csv_path)

    return per_seed


def save_aggregate(out_root: Path) -> None:
    """Tulis aggregate.csv dan matrix_summary.json lintas seed.

    Args:
        out_root (Path): Direktori akar hasil matriks.
    """
    per_seed = collect_seed_results(out_root)

    if not per_seed:
        print(f"Tidak ada {SEED_DIR_PREFIX}*/results.csv di {out_root}")
        return

    # Latency adalah satu pengukuran per matriks. Terapkan juga ke CSV seed
    # yang dibuat di sesi lain (mis. hasil yang diunduh dari Colab), agar H3
    # tidak membandingkan angka dari sesi pengukuran berbeda.
    latency = None
    latency_path = out_root / LATENCY_FILE

    if latency_path.exists():
        latency = json.loads(latency_path.read_text())

        for rows in per_seed.values():
            apply_latency(rows, latency)

    report = evaluate_replicated_hypotheses(per_seed)
    labels = report["seeds"]
    aggregate_rows = []

    for mean_row, sd_row in zip(report["mean_rows"], report["sd_rows"]):
        row = {
            "id": mean_row["id"],
            "desc": mean_row.get("desc", ""),
            "attention": mean_row.get("attention"),
            "seeds": len(labels),
            "total_params": mean_row.get("total_params"),
        }

        for metric in AGGREGATE_METRICS:
            row[f"{metric}_mean"] = mean_row.get(metric)
            row[f"{metric}_sd"] = sd_row.get(metric)

        aggregate_rows.append(row)

    write_results_csv(out_root / "aggregate.csv", aggregate_rows)

    matrix = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_seeds": report["n_seeds"],
        "seeds": labels,
        "aggregation_rule": {
            "metrics": "rata-rata aritmetika antar seed",
            "verdict": "kriteria H1-H3 yang sama pada baris rata-rata",
            "acceptance": (
                "agregat accepted dan minimal "
                f"{report['required_support']}/{len(labels)} seed accepted"
            ),
            "note": "E5 tetap deskriptif; toleransi dan varian utama tidak diubah",
        },
        "experiments": report["mean_rows"],
        "dispersion": report["sd_rows"],
        "hypotheses": report["hypotheses"],
        "aggregate_hypotheses": report["aggregate_hypotheses"],
        "per_seed_hypotheses": report["per_seed_hypotheses"],
        "per_seed_experiments": per_seed,
    }

    if latency is not None:
        matrix["latency_benchmark"] = latency

    output_path = out_root / "matrix_summary.json"
    output_path.write_text(json.dumps(matrix, indent=2))

    print(f"Saved {output_path}")
    print(f"Agregat {len(labels)} seed: {', '.join(labels)}")

    for name, verdict in report["hypotheses"].items():
        print(f"{name}: {verdict['status']} ({verdict['support']})")


def build_command(experiment: dict, save_dir: Path) -> list:
    """Susun perintah subprocess main.py dari dict eksperimen.

    Args:
        experiment (dict): Config eksperimen hasil merge YAML.
        save_dir (Path): Direktori keluaran eksperimen ini.

    Returns:
        list: Perintah siap `subprocess.run` (melewati nilai None).
    """
    # -u: paksa stdout child unbuffered agar log epoch muncul real-time
    # di terminal saat run_matrix memantau proses.
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "main.py")]
    for flag, key in MAIN_FLAGS:
        value = experiment.get(key)

        if value is None:
            continue

        cmd.extend([flag, str(value)])

    cmd.extend(["--save-dir", str(save_dir), "--gradcam"])

    if experiment["denoise"]:
        cmd.append("--denoise")

    if not experiment.get("pos_weight", True):
        cmd.append("--no-pos-weight")

    return cmd


def stream_subprocess(cmd: list, log_path: Path) -> tuple[int, str]:
    """Jalankan subprocess dengan keluaran di-stream live ke terminal.

    stdout dan stderr digabung, lalu dibaca per potongan byte (bukan per
    baris) supaya pembaruan progress bar tqdm yang memakai carriage return
    ikut tampil real-time. Seluruh keluaran tetap tersimpan utuh di berkas
    log, termasuk saat proses gagal atau dihentikan.

    Args:
        cmd (list): Perintah siap dieksekusi.
        log_path (Path): Berkas log tujuan.

    Returns:
        tuple: (returncode, seluruh teks keluaran).

    Raises:
        KeyboardInterrupt: Setelah child dihentikan dan log parsial ditulis.
    """
    chunks: list[str] = []
    interrupted = False
    returncode = -1

    with subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        assert proc.stdout is not None
        stdout = cast(BufferedReader, proc.stdout)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        try:
            while True:
                # read1 mengembalikan byte yang sudah tersedia tanpa menunggu
                # buffer 4096 penuh, sehingga progress bar dan log langsung
                # tampil saat child masih berjalan.
                raw = stdout.read1(4096)

                if not raw:
                    break

                text = decoder.decode(raw)

                if text:
                    chunks.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            interrupted = True
        else:
            returncode = proc.wait()
        finally:
            tail = decoder.decode(b"", final=True)

            if tail:
                chunks.append(tail)
                sys.stdout.write(tail)
                sys.stdout.flush()

    output = "".join(chunks)
    log_path.write_text(output)

    if interrupted:
        raise KeyboardInterrupt

    return returncode, output


def run_experiment(experiment: dict, out_root: Path) -> dict:
    """Jalankan satu eksperimen sebagai subprocess dengan log live.

    Args:
        experiment (dict): Config eksperimen hasil merge YAML.
        out_root (Path): Direktori akar hasil matriks.

    Returns:
        dict: Baris hasil (id, config, metrik, parameter, latency,
            returncode) untuk results.csv.
    """
    save_dir = out_root / experiment["id"]
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(experiment, save_dir)
    log_path = save_dir / "stdout.log"

    print(f"{experiment['id']}: {experiment['desc']}", flush=True)
    returncode, output = stream_subprocess(cmd, log_path)

    result = {
        "id": experiment["id"],
        "desc": experiment["desc"],
        "denoise": experiment["denoise"],
        "enhance": experiment["enhance"],
        "segment": experiment["segment"],
        "input_mode": experiment["input_mode"],
        "attention": experiment["attention"],
        "pos_weight": experiment.get("pos_weight", True),
        "seed": experiment.get("seed"),
        "returncode": returncode,
    }

    result.update(parse_metrics(output))
    total_params, infer_ms = model_stats(experiment["attention"])
    result["total_params"] = total_params
    result["infer_ms_cpu"] = infer_ms
    status = "OK" if returncode == 0 else "FAIL"

    print(f"{experiment['id']} status: {status}", flush=True)

    return result


def save_matrix_summary(
    out_root: Path, rows: list, latency: dict | None = None
) -> None:
    """Tulis matrix_summary.json: baris eksperimen + verdict + run summary.

    Args:
        out_root (Path): Direktori akar hasil matriks.
        rows (list): Baris hasil tiap eksperimen.
        latency (dict | None): Payload benchmark latency bersama, bila ada.
    """
    summaries = {}
    for row in rows:
        summary_path = out_root / row["id"] / "run_summary.json"
        if summary_path.exists():
            summaries[row["id"]] = json.loads(summary_path.read_text())

    matrix = {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiments": rows,
        "hypotheses": evaluate_hypotheses(rows),
        "run_summaries": summaries,
    }

    if latency is not None:
        matrix["latency_benchmark"] = latency

    output_path = out_root / "matrix_summary.json"
    output_path.write_text(json.dumps(matrix, indent=2))

    print(f"Saved {output_path}")


def run_selected(experiments: list, out_root: Path, seed: int | None = None) -> list:
    """Jalankan eksperimen terpilih pada satu seed.

    Args:
        experiments (list): Eksperimen hasil seleksi.
        out_root (Path): Direktori hasil untuk seed ini.
        seed (int | None): Seed yang dipaksakan; None memakai nilai YAML.

    Returns:
        list: Baris hasil tiap eksperimen.
    """
    rows = []

    for experiment in experiments:
        item = dict(experiment)

        if seed is not None:
            item["seed"] = seed

        rows.append(run_experiment(item, out_root))

    return rows


def main() -> None:
    """Orkestrasi matriks: muat YAML, jalankan eksperimen, tulis CSV+JSON."""
    parser = argparse.ArgumentParser(description="Run PCD+ACM experiment matrix")
    parser.add_argument("--config", default="experiments/experiments.yml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Daftar seed koma-pisah (mis. 42,43,44). Layout hasil menjadi "
            "seed-<n>/E* dan agregasi ditulis ke aggregate.csv. Bila hanya "
            "sebagian seed dijalankan, agregasi memakai semua seed-*/ yang ada."
        ),
    )
    parser.add_argument("--gradcam-samples", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--only", default="", help="Comma-separated subset, e.g. E1,E2")
    parser.add_argument(
        "--latency-rounds",
        type=int,
        default=5,
        help="Cacah putaran benchmark latency (default: 5)",
    )
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=20,
        help="Cacah forward per putaran benchmark (default: 20)",
    )
    parser.add_argument(
        "--remeasure-latency",
        action="store_true",
        help="Abaikan cache latency.json dan ukur ulang",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Hanya agregasi seed-*/results.csv yang sudah ada, tanpa training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the main.py commands without running them",
    )
    args = parser.parse_args()

    if args.seeds and args.seed is not None:
        parser.error("Gunakan --seeds atau --seed, bukan keduanya")

    try:
        seeds = parse_seed_list(args.seeds) if args.seeds else None
    except ValueError as error:
        parser.error(str(error))

    config_path = PROJECT_ROOT / args.config
    _, experiments = load_config(config_path)
    experiments = resolve_overrides(args, experiments)
    selected = select_experiments(experiments, args.only)
    out_root = PROJECT_ROOT / selected[0]["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        save_aggregate(out_root)

        return

    if seeds is None:
        runs = [(None, out_root)]
    else:
        runs = [(seed, out_root / f"{SEED_DIR_PREFIX}{seed}") for seed in seeds]

    if args.dry_run:
        for seed, run_root in runs:
            prefix = f"[seed {seed}] " if seed is not None else ""

            for experiment in selected:
                item = dict(experiment)

                if seed is not None:
                    item["seed"] = seed

                command = build_command(item, run_root / experiment["id"])
                print(f"{prefix}{shlex.join(command)}")

        return

    attention_types = sorted({experiment["attention"] for experiment in selected})
    written = []

    for seed, run_root in runs:
        if seed is not None:
            print(f"Seed {seed}", flush=True)

        written.append((run_root, run_selected(selected, run_root, seed)))

    latency = load_or_measure_latency(
        out_root,
        attention_types,
        rounds=args.latency_rounds,
        repeats=args.latency_repeats,
        remeasure=args.remeasure_latency,
    )

    for run_root, rows in written:
        apply_latency(rows, latency)
        write_results_csv(run_root / "results.csv", rows)

    if seeds is None:
        save_matrix_summary(out_root, written[0][1], latency)
    else:
        save_aggregate(out_root)


if __name__ == "__main__":
    main()
