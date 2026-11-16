"""Test pos_weight BCE: kebenaran rasio, integrasi train(), dan jalur gagal.

Jalankan dari root repo:

    python -m unittest tests.test_pos_weight -v
"""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.dataset import TransformedSubset
from src.run_summary import SUMMARY_METRICS, save_run_summary
from src.train import build_criterion, compute_pos_weight, train


class _TinyNet(nn.Module):
    """Model mini untuk uji integrasi train() di CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FakeBaseDataset(Dataset):
    """Dataset mini berbasis .samples untuk uji jalur TransformedSubset."""

    def __init__(self, samples: list) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        return self.samples[index]


class ComputePosWeightTests(unittest.TestCase):
    def test_balanced_gives_one(self) -> None:
        features = torch.randn(10, 4)
        labels = torch.tensor([0.0, 1.0] * 5)
        loader = DataLoader(TensorDataset(features, labels))
        self.assertAlmostEqual(compute_pos_weight(loader), 1.0)

    def test_imbalanced_ratio_correct(self) -> None:
        # 9 negatif, 3 positif -> pos_weight = 3.0.
        features = torch.randn(12, 4)
        labels = torch.tensor([0.0] * 9 + [1.0] * 3)
        loader = DataLoader(TensorDataset(features, labels))
        self.assertAlmostEqual(compute_pos_weight(loader), 3.0)

    def test_majority_positive_gives_below_one(self) -> None:
        # Kasus nyata: label 1 (infected) mayoritas -> pos_weight < 1.
        features = torch.randn(12, 4)
        labels = torch.tensor([1.0] * 9 + [0.0] * 3)
        loader = DataLoader(TensorDataset(features, labels))
        self.assertAlmostEqual(compute_pos_weight(loader), 1 / 3)

    def test_transformed_subset_reads_only_indices(self) -> None:
        # Hanya label indeks subset yang dihitung (tidak seluruh dataset).
        base = _FakeBaseDataset([(0, 1), (1, 1), (2, 0), (3, 0)])
        subset = TransformedSubset(base, indices=[0, 1, 2], transform=None)
        loader = DataLoader(subset)
        self.assertAlmostEqual(compute_pos_weight(loader), 0.5)

    def test_single_class_raises(self) -> None:
        features = torch.randn(6, 4)
        labels = torch.zeros(6)
        with self.assertRaises(ValueError):
            compute_pos_weight(DataLoader(TensorDataset(features, labels)))


class BuildCriterionTests(unittest.TestCase):
    """Regresi device: pos_weight wajib ikut device model dan logits."""

    def setUp(self) -> None:
        features = torch.randn(12, 4)
        labels = torch.tensor([0.0] * 9 + [1.0] * 3)
        self.loader = DataLoader(TensorDataset(features, labels))

    def test_pos_weight_follows_requested_device(self) -> None:
        value, criterion = build_criterion(self.loader, torch.device("meta"))

        self.assertIsNotNone(value)
        self.assertAlmostEqual(cast(float, value), 3.0)
        self.assertIsNotNone(criterion.pos_weight)
        self.assertEqual(criterion.pos_weight.device.type, "meta")

        logits = torch.empty(2, device="meta")
        target = torch.empty(2, device="meta")
        loss = criterion(logits, target)
        self.assertEqual(loss.device.type, "meta")

    def test_disabled_returns_plain_criterion(self) -> None:
        value, criterion = build_criterion(
            self.loader, torch.device("cpu"), pos_weight=False
        )

        self.assertIsNone(value)
        self.assertIsNone(criterion.pos_weight)

    def test_value_matches_compute_pos_weight(self) -> None:
        value, _ = build_criterion(self.loader, torch.device("cpu"))

        self.assertIsNotNone(value)
        self.assertAlmostEqual(cast(float, value), compute_pos_weight(self.loader))


class TrainPosWeightIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        rng = torch.Generator().manual_seed(0)
        self.features = torch.randn(40, 4, generator=rng)
        self.labels = torch.tensor([1.0] * 30 + [0.0] * 10)  # pos_weight 1/3

        loader = DataLoader(TensorDataset(self.features, self.labels), batch_size=8)
        self.train_loader = loader
        self.val_loader = loader
        self.config = {
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "patience": 5,
            "batch_size": 8,
            "save_dir": self.tmp.name,
        }

    def _run_train(self) -> dict:
        return train(_TinyNet(), self.train_loader, self.val_loader, self.config)

    def test_train_applies_pos_weight_and_records_it(self) -> None:
        history = self._run_train()

        self.assertTrue(
            all(torch.isfinite(torch.tensor(v)) for v in history["train_loss"])
        )

        checkpoint = torch.load(
            Path(self.tmp.name) / "best_model.pth", weights_only=True
        )
        self.assertAlmostEqual(checkpoint["pos_weight"], 1 / 3, places=5)

    def test_train_without_pos_weight_records_none(self) -> None:
        self.config["pos_weight"] = False
        self._run_train()

        checkpoint = torch.load(
            Path(self.tmp.name) / "best_model.pth", weights_only=True
        )
        self.assertIsNone(checkpoint["pos_weight"])

    def test_run_summary_records_pos_weight(self) -> None:
        self._run_train()

        args = argparse.Namespace(
            attention="self_attention", dropout=0.3, no_pretrained=True
        )
        config = {
            **self.config,
            "epochs": self.config["epochs"],
            "pos_weight": True,
        }
        history = {
            "train_loss": [0.7],
            "train_acc": [0.5],
            "val_loss": [0.7],
            "val_acc": [0.5],
        }
        test_metrics = {key: 0.5 for key in SUMMARY_METRICS}

        save_run_summary(
            self.tmp.name,
            args,
            config,
            {"train": 40, "val": 8, "test": 8},
            {"total_params": 5, "trainable_params": 5},
            history,
            test_metrics,
            torch.device("cpu"),
        )

        summary = json.loads((Path(self.tmp.name) / "run_summary.json").read_text())
        self.assertEqual(summary["loss"]["type"], "BCEWithLogitsLoss")
        self.assertAlmostEqual(summary["loss"]["pos_weight"], 1 / 3, places=5)


if __name__ == "__main__":
    unittest.main()
