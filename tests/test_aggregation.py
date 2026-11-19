"""Uji agregasi multi-seed, aturan dukungan mayoritas, dan benchmark latency.

Jalankan dari root repo:

    python -m unittest tests.test_aggregation -v
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT, _ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_matrix  # type: ignore[reportMissingImports]
from hypotheses import (  # type: ignore[reportMissingImports]
    aggregate_seed_rows,
    evaluate_replicated_hypotheses,
)


def _row(
    experiment_id: str,
    *,
    accuracy: float = 0.97,
    auc: float = 0.99,
    f1: float | None = None,
    total_params: int = 1000,
    infer_ms_cpu: float = 10.0,
) -> dict:
    """Baris hasil sintetis dengan kolom yang dipakai evaluator."""
    return {
        "id": experiment_id,
        "desc": f"eksperimen {experiment_id}",
        "attention": "self_attention",
        "accuracy": accuracy,
        "auc": auc,
        "f1": accuracy if f1 is None else f1,
        "total_params": total_params,
        "infer_ms_cpu": infer_ms_cpu,
    }


class _TinyModel(torch.nn.Module):
    """Model kecil agar benchmark latency bisa diuji tanpa DenseNet-121."""

    def __init__(self, pretrained: bool = False, attention_type: str = "") -> None:
        super().__init__()
        self.pretrained = pretrained
        self.attention_type = attention_type
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(inputs).flatten(1))


class SeedListTests(unittest.TestCase):
    def test_parses_unique_seeds_in_order(self) -> None:
        self.assertEqual(run_matrix.parse_seed_list(" 42, 43 ,44 "), [42, 43, 44])

    def test_ignores_empty_fragments(self) -> None:
        self.assertEqual(run_matrix.parse_seed_list("42,,43,"), [42, 43])

    def test_empty_text_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_matrix.parse_seed_list("")

    def test_non_integer_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_matrix.parse_seed_list("42,abc")

    def test_negative_seed_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_matrix.parse_seed_list("-1")

    def test_duplicate_seed_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_matrix.parse_seed_list("42,42")


class AggregateRowsTests(unittest.TestCase):
    def test_mean_and_sample_sd_across_seeds(self) -> None:
        per_seed = {
            "42": [_row("E1", accuracy=0.98, auc=0.99)],
            "43": [_row("E1", accuracy=0.96, auc=0.97)],
        }

        mean_rows, sd_rows = aggregate_seed_rows(per_seed)

        self.assertAlmostEqual(mean_rows[0]["accuracy"], 0.97)
        self.assertAlmostEqual(mean_rows[0]["auc"], 0.98)
        self.assertAlmostEqual(sd_rows[0]["accuracy"], 0.0141421356, places=6)

    def test_single_seed_has_zero_sd(self) -> None:
        per_seed = {"42": [_row("E1", accuracy=0.98, auc=0.99)]}

        mean_rows, sd_rows = aggregate_seed_rows(per_seed)

        self.assertEqual(mean_rows[0]["accuracy"], 0.98)
        self.assertEqual(sd_rows[0]["accuracy"], 0.0)

    def test_missing_experiment_in_one_seed_is_ignored(self) -> None:
        per_seed = {
            "42": [_row("E1"), _row("E2")],
            "43": [_row("E1")],
        }

        mean_rows, _ = aggregate_seed_rows(per_seed)

        self.assertEqual([row["id"] for row in mean_rows], ["E1", "E2"])
        self.assertEqual(mean_rows[1]["accuracy"], 0.97)

    def test_empty_input_returns_empty_lists(self) -> None:
        self.assertEqual(aggregate_seed_rows({}), ([], []))


class ReplicatedVerdictTests(unittest.TestCase):
    @staticmethod
    def _h2_rows(f1_e4: float, auc_e4: float) -> list:
        return [
            _row("E3", accuracy=0.97, auc=0.99, f1=0.97),
            _row("E4", accuracy=0.97, auc=auc_e4, f1=f1_e4),
        ]

    def test_majority_support_accepts_h2(self) -> None:
        per_seed = {
            "42": self._h2_rows(0.98, 0.995),
            "43": self._h2_rows(0.96, 0.991),
            "44": self._h2_rows(0.975, 0.992),
        }

        report = evaluate_replicated_hypotheses(per_seed)

        self.assertEqual(report["required_support"], 2)
        self.assertEqual(report["hypotheses"]["H2"]["status"], "accepted")
        self.assertEqual(report["hypotheses"]["H2"]["support"], "2/3")

    def test_aggregate_accepted_but_single_seed_support_rejected(self) -> None:
        # Dua seed gagal di F1, tetapi rata-ratanya masih di dalam toleransi.
        per_seed = {
            "42": self._h2_rows(0.98, 0.995),
            "43": self._h2_rows(0.96, 0.992),
            "44": self._h2_rows(0.96, 0.992),
        }

        report = evaluate_replicated_hypotheses(per_seed)

        self.assertEqual(report["aggregate_hypotheses"]["H2"]["status"], "accepted")
        self.assertEqual(report["hypotheses"]["H2"]["status"], "not_accepted")
        self.assertEqual(report["hypotheses"]["H2"]["support"], "1/3")

    def test_minority_support_rejects_even_when_mean_passes(self) -> None:
        per_seed = {
            "42": self._h2_rows(0.96, 0.992),
            "43": self._h2_rows(0.96, 0.992),
            "44": self._h2_rows(0.98, 0.995),
        }

        report = evaluate_replicated_hypotheses(per_seed)

        self.assertEqual(report["hypotheses"]["H2"]["status"], "not_accepted")

    def test_single_seed_requires_its_own_verdict(self) -> None:
        per_seed = {"42": self._h2_rows(0.96, 0.992)}

        report = evaluate_replicated_hypotheses(per_seed)

        self.assertEqual(report["required_support"], 1)
        self.assertEqual(report["hypotheses"]["H2"]["status"], "not_accepted")

    def test_hypotheses_without_data_are_skipped(self) -> None:
        per_seed = {"42": self._h2_rows(0.98, 0.995)}

        report = evaluate_replicated_hypotheses(per_seed)

        self.assertNotIn("H1", report["hypotheses"])
        self.assertNotIn("H3", report["hypotheses"])


class LatencyBenchmarkTests(unittest.TestCase):
    def test_alternating_benchmark_reports_median_and_params(self) -> None:
        stats = run_matrix.benchmark_latency(
            ["self_attention", "cbam"],
            rounds=2,
            repeats=2,
            model_factory=_TinyModel,
        )

        self.assertEqual(set(stats), {"self_attention", "cbam"})

        for attention in ("self_attention", "cbam"):
            entry = stats[attention]
            self.assertEqual(len(entry["samples_ms"]), 2)
            self.assertLessEqual(entry["min_ms"], entry["median_ms"])
            self.assertLessEqual(entry["median_ms"], entry["max_ms"])
            self.assertEqual(entry["total_params"], 4)

    def test_invalid_rounds_raise(self) -> None:
        with self.assertRaises(ValueError):
            run_matrix.benchmark_latency(["cbam"], rounds=0, model_factory=_TinyModel)

    def test_invalid_repeats_raise(self) -> None:
        with self.assertRaises(ValueError):
            run_matrix.benchmark_latency(["cbam"], repeats=0, model_factory=_TinyModel)

    def test_apply_latency_overrides_row_and_leaves_unknown_attention(self) -> None:
        rows = [
            {"attention": "cbam", "infer_ms_cpu": 999.0, "total_params": 1},
            {"attention": "se_net", "infer_ms_cpu": 5.0, "total_params": 2},
        ]
        latency = {
            "per_attention": {"cbam": {"median_ms": 12.5, "total_params": 42}}
        }

        run_matrix.apply_latency(rows, latency)

        self.assertEqual(rows[0]["infer_ms_cpu"], 12.5)
        self.assertEqual(rows[0]["total_params"], 42)
        self.assertEqual(rows[1]["infer_ms_cpu"], 5.0)
        self.assertEqual(rows[1]["total_params"], 2)


class RunSelectedTests(unittest.TestCase):
    def test_seed_is_injected_into_every_experiment(self) -> None:
        experiment = {"id": "E1", "desc": "baseline"}

        with mock.patch.object(
            run_matrix, "run_experiment", return_value={"id": "E1"}
        ) as mocked:
            run_matrix.run_selected([experiment], Path("/tmp"), seed=43)

        passed_experiment = mocked.call_args.args[0]
        self.assertEqual(passed_experiment["seed"], 43)
        self.assertNotIn("seed", experiment)

    def test_none_seed_keeps_yaml_value(self) -> None:
        experiment = {"id": "E1", "seed": 42}

        with mock.patch.object(
            run_matrix, "run_experiment", return_value={"id": "E1"}
        ) as mocked:
            run_matrix.run_selected([experiment], Path("/tmp"))

        passed_experiment = mocked.call_args.args[0]
        self.assertEqual(passed_experiment["seed"], 42)


class AggregateFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_root = Path(self.tmp.name) / "results"
        self.out_root.mkdir(parents=True)

    def _write_seed(self, label: str, rows: list) -> None:
        seed_dir = self.out_root / f"{run_matrix.SEED_DIR_PREFIX}{label}"
        seed_dir.mkdir(parents=True)
        run_matrix.write_results_csv(seed_dir / "results.csv", rows)

    def test_save_aggregate_writes_mean_sd_and_verdicts(self) -> None:
        base = {
            "id": "E3",
            "desc": "denoise + clahe",
            "attention": "self_attention",
            "accuracy": 0.97,
            "auc": 0.99,
            "f1": 0.97,
            "total_params": 1000,
            "infer_ms_cpu": 10.0,
        }
        seed42 = [
            dict(base),
            {
                **base,
                "id": "E4",
                "desc": "roi otsu",
                "auc": 0.995,
                "f1": 0.98,
            },
        ]
        seed43 = [
            dict(base),
            {**base, "id": "E4", "desc": "roi otsu", "auc": 0.992, "f1": 0.975},
        ]

        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            self._write_seed("42", seed42)
            self._write_seed("43", seed43)
            run_matrix.save_aggregate(self.out_root)

        output = buffer.getvalue()
        self.assertIn("Agregat 2 seed: 42, 43", output)
        self.assertIn("H2: accepted (2/2)", output)

        matrix = json.loads((self.out_root / "matrix_summary.json").read_text())
        self.assertEqual(matrix["n_seeds"], 2)
        self.assertEqual(matrix["seeds"], ["42", "43"])
        self.assertEqual(matrix["hypotheses"]["H2"]["status"], "accepted")
        self.assertEqual(matrix["hypotheses"]["H2"]["support"], "2/2")

        header = (self.out_root / "aggregate.csv").read_text().splitlines()[0]
        self.assertIn("accuracy_mean", header)
        self.assertIn("accuracy_sd", header)
        self.assertIn("infer_ms_cpu_mean", header)

    def test_saved_latency_is_applied_to_imported_seed_csv(self) -> None:
        row = {
            "id": "E6",
            "desc": "cbam",
            "attention": "cbam",
            "accuracy": 0.96,
            "auc": 0.99,
            "f1": 0.97,
            "total_params": 1,
            "infer_ms_cpu": 999.0,
        }

        with contextlib.redirect_stdout(io.StringIO()):
            self._write_seed("42", [row])
            (self.out_root / run_matrix.LATENCY_FILE).write_text(
                json.dumps(
                    {
                        "per_attention": {
                            "cbam": {"median_ms": 12.5, "total_params": 42}
                        }
                    }
                )
            )
            run_matrix.save_aggregate(self.out_root)

        matrix = json.loads((self.out_root / "matrix_summary.json").read_text())
        imported = matrix["per_seed_experiments"]["42"][0]
        self.assertEqual(imported["infer_ms_cpu"], 12.5)
        self.assertEqual(imported["total_params"], 42)

    def test_save_matrix_summary_single_run_prints_verdicts(self) -> None:
        rows = [_row("E3"), _row("E4", f1=0.98, auc=0.995)]

        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            run_matrix.save_matrix_summary(self.out_root, rows)

        matrix = json.loads((self.out_root / "matrix_summary.json").read_text())
        self.assertEqual(matrix["hypotheses"]["H2"]["status"], "accepted")

        output = buffer.getvalue()
        self.assertIn("Saved", output)

    def test_save_aggregate_without_seed_dirs_does_not_write(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            run_matrix.save_aggregate(self.out_root)

        self.assertFalse((self.out_root / "matrix_summary.json").exists())
        self.assertFalse((self.out_root / "aggregate.csv").exists())


if __name__ == "__main__":
    unittest.main()
