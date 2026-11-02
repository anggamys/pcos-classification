"""Test jalur gagal (negative tests): data rusak, konfigurasi salah, input ekstrem.

Tujuannya memastikan kegagalan bersifat LOUD (error jelas, tidak lanjut
diam-diam menghasilkan hasil salah), bukan hanya memverifikasi jalur sukses.

Jalankan dari root repo:

    python -m unittest tests.test_failure_paths -v
"""

import tempfile
import unittest
from collections.abc import Sized
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.dataset import PCOSDataset, create_dataloaders
from src.gradcam import GradCAM, visualize_gradcam
from src.models.attention import SelfAttention, TransformerAttention
from src.models.densenet121 import DenseNet121Attention
from src.preprocessing.quality import enhance, enhance_clahe, psnr, ssim
from src.preprocessing.segmentation import overlay_mask, segment


def make_image_dir(directory: Path, count: int, size: int = 32) -> None:
    """Buat direktori berisi citra .jpg acak (unik antar-citra) untuk dataset uji.

    Args:
        directory (Path): Direktori tujuan (dibuat bila belum ada).
        count (int): Jumlah citra.
        size (int): Sisi persegi citra (default 32).
    """
    directory.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        # Seed unik per (nama dir, indeks) agar tidak ada duplikat md5
        # antar-direktori yang menguji jalur dedup tanpa sengaja.
        rng = np.random.default_rng([index, *[ord(c) for c in directory.name]])
        arr = (rng.random((size, size)) * 255).astype(np.uint8)
        Image.fromarray(arr).convert("RGB").save(directory / f"img_{index}.jpg")


class DatasetFailurePaths(unittest.TestCase):
    """Dataset rusak atau konfigurasi split salah harus error jelas."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_missing_root_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            PCOSDataset(self.root / "tidak-ada")

    def test_missing_class_subdir_raises(self) -> None:
        make_image_dir(self.root / "infected", 2)
        with self.assertRaises(FileNotFoundError):
            PCOSDataset(self.root)

    def test_empty_class_subdir_raises(self) -> None:
        (self.root / "noninfected").mkdir(parents=True)
        make_image_dir(self.root / "infected", 2)
        with self.assertRaises(FileNotFoundError):
            PCOSDataset(self.root)

    def test_non_jpg_only_raises(self) -> None:
        # Hanya .png di noninfected: glob *.jpg kosong -> harus error, bukan
        # dataset diam-diam satu kelas.
        (self.root / "noninfected").mkdir(parents=True)
        arr = np.zeros((32, 32), dtype=np.uint8)
        Image.fromarray(arr).convert("RGB").save(self.root / "noninfected" / "a.png")
        make_image_dir(self.root / "infected", 2)

        with self.assertRaises(FileNotFoundError):
            PCOSDataset(self.root)

    def test_corrupt_image_raises_loudly(self) -> None:
        # File korup tidak di-swallow: PIL melempar error saat dibaca.
        make_image_dir(self.root / "noninfected", 1)
        make_image_dir(self.root / "infected", 2)
        (self.root / "infected" / "img_0.jpg").write_bytes(b"bukan-jpg")

        dataset = PCOSDataset(self.root, dedupe=False)
        # Urutan sampel: noninfected dulu, lalu infected (img_0 korup).
        with self.assertRaises(UnidentifiedImageError):
            dataset[1]

    def test_dedup_vanishing_class_raises(self) -> None:
        # Semua citra infected salinan byte-identik dari noninfected ->
        # dedup keep-first memusnahkan kelas infected -> harus error,
        # bukan dataset satu kelas yang dilatih senyap.
        make_image_dir(self.root / "noninfected", 2)
        (self.root / "infected").mkdir(parents=True)

        for source in sorted((self.root / "noninfected").glob("*.jpg")):
            (self.root / "infected" / source.name).write_bytes(source.read_bytes())

        with self.assertRaises(ValueError) as ctx:
            PCOSDataset(self.root)
        self.assertIn("kelas", str(ctx.exception))

    def test_valid_minimal_dataset_works(self) -> None:
        make_image_dir(self.root / "noninfected", 2)
        make_image_dir(self.root / "infected", 2)

        dataset = PCOSDataset(self.root)
        self.assertEqual(len(dataset), 4)
        self.assertEqual({label for _, label in dataset.samples}, {0, 1})


class SplitRatioFailurePaths(unittest.TestCase):
    """Rasio split tidak valid atau dataset terlalu kecil harus error."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        make_image_dir(self.root / "noninfected", 4)
        make_image_dir(self.root / "infected", 4)

    def test_ratios_exceeding_one_raise(self) -> None:
        # 0.8 + 0.3 = 1.1 -> test set kosong (dulu senyap: test=0).
        with self.assertRaises(ValueError):
            create_dataloaders(self.root, train_ratio=0.8, val_ratio=0.3, num_workers=0)

    def test_train_ratio_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_dataloaders(self.root, train_ratio=1.0, val_ratio=0.0, num_workers=0)

    def test_negative_ratio_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_dataloaders(
                self.root, train_ratio=0.7, val_ratio=-0.1, num_workers=0
            )

    def test_dataset_too_small_raises(self) -> None:
        # total=5 -> val=0 -> split invalid (dulu senyap: val kosong).
        small = self.root / "kecil"
        make_image_dir(small / "noninfected", 3)
        make_image_dir(small / "infected", 2)

        with self.assertRaises(ValueError):
            create_dataloaders(small, train_ratio=0.7, val_ratio=0.15, num_workers=0)

    def test_valid_ratios_produce_nonempty_splits(self) -> None:
        train, val, test = create_dataloaders(
            self.root, train_ratio=0.7, val_ratio=0.15, num_workers=0
        )
        self.assertEqual(
            (
                len(cast(Sized, train.dataset)),
                len(cast(Sized, val.dataset)),
                len(cast(Sized, test.dataset)),
            ),
            (5, 1, 2),
        )


class PreprocessingFailurePaths(unittest.TestCase):
    """Mask salah ukuran, tiles ekstrem, dan metrik beda ukuran."""

    def test_overlay_mask_shape_mismatch_raises_clearly(self) -> None:
        image = Image.new("RGB", (10, 10), 128)
        with self.assertRaises(ValueError) as ctx:
            overlay_mask(image, np.zeros((5, 5), dtype=bool))

        # Pesan harus menyebut mask dan citra, bukan "images do not match".
        self.assertIn("mask", str(ctx.exception))
        self.assertIn("citra", str(ctx.exception))

    def test_overlay_mask_valid_shapes_work(self) -> None:
        image = Image.new("RGB", (10, 10), 128)
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:6, 3:6] = True
        result = overlay_mask(image, mask)
        self.assertEqual(result.size, (10, 10))

    def test_clahe_zero_tiles_raises(self) -> None:
        image = Image.new("RGB", (32, 32), 128)
        with self.assertRaises(ValueError):
            enhance_clahe(image, tiles=0)

    def test_clahe_oversized_tiles_raise(self) -> None:
        # tiles=1000 pada citra 128px -> 1px tiles -> kontras jenuh (dulu
        # senyap menghasilkan citra semua-255).
        image = Image.new("RGB", (128, 128), 128)
        with self.assertRaises(ValueError):
            enhance_clahe(image, tiles=1000)

    def test_clahe_reasonable_tiles_still_work(self) -> None:
        rng = np.random.default_rng(1)
        arr = (rng.random((64, 64)) * 50 + 100).astype(np.uint8)
        image = Image.fromarray(arr).convert("RGB")

        result = enhance_clahe(image, tiles=8)
        self.assertEqual(result.size, (64, 64))
        self.assertGreater(np.array(result.convert("L")).std(), 0)

    def test_psnr_shape_mismatch_raises(self) -> None:
        # Dulu senyap me-resize; sekarang error agar metrik tidak menyesatkan.
        small = Image.new("RGB", (8, 8), 0)
        large = Image.new("RGB", (16, 16), 0)
        with self.assertRaises(ValueError):
            psnr(small, large)

    def test_ssim_shape_mismatch_raises(self) -> None:
        small = Image.new("RGB", (8, 8), 0)
        large = Image.new("RGB", (16, 16), 0)
        with self.assertRaises(ValueError):
            ssim(small, large)

    def test_metrics_identical_images(self) -> None:
        image = Image.fromarray(np.full((32, 32), 128, dtype=np.uint8)).convert("RGB")
        self.assertEqual(psnr(image, image), float("inf"))
        self.assertGreater(ssim(image, image), 0.99)

    def test_unknown_enhance_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            enhance(Image.new("RGB", (8, 8), 128), method="equalize")

    def test_unknown_segment_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            segment(Image.new("RGB", (8, 8), 128), method="kmeans")


class AttentionFailurePaths(unittest.TestCase):
    """Input terlalu kecil tidak boleh kembali degenerasi N=1 senyap."""

    def test_selfattention_single_token_raises(self) -> None:
        attention = SelfAttention(embed_dim=64, num_heads=4)
        with self.assertRaises(ValueError):
            attention(torch.randn(2, 1, 64))

    def test_transformer_single_token_raises(self) -> None:
        attention = TransformerAttention(embed_dim=64, num_heads=4, num_layers=1)
        with self.assertRaises(ValueError):
            attention(torch.randn(2, 1, 64))

    def test_selfattention_multi_token_works(self) -> None:
        attention = SelfAttention(embed_dim=64, num_heads=4)
        output = attention(torch.randn(2, 49, 64))
        self.assertEqual(tuple(output.shape), (2, 49, 64))

    def test_tiny_image_raises_instead_of_degenerate_attention(self) -> None:
        # Input 32px -> peta fitur 1x1 -> N=1; dulu senyap (attention mati),
        # kini error jelas.
        model = DenseNet121Attention(pretrained=False, attention_type="self_attention")
        model.eval()
        with self.assertRaises(ValueError), torch.no_grad():
            model(torch.randn(1, 3, 32, 32))

    def test_model_small_but_valid_input_works(self) -> None:
        # 128px -> fitur 4x4 -> N=16; harus tetap jalan.
        model = DenseNet121Attention(pretrained=False, attention_type="self_attention")
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 128, 128))
        self.assertEqual(tuple(output.shape), (1, 1))


class _MockBackboneModel(torch.nn.Module):
    """Model kecil dengan struktur backbone yang dikenali Grad-CAM.

    Args:
        outputs (int): Cacah nilai keluaran (1 = biner satu-logit).
    """

    def __init__(self, outputs: int = 1) -> None:
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.features = torch.nn.Module()
        self.denseblock4 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.backbone.features.denseblock4 = self.denseblock4
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(4, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.denseblock4(x))
        return self.fc(self.pool(x).flatten(1))


class GradCAMFailurePaths(unittest.TestCase):
    """Grad-CAM harus menolak masukan tidak valid secara eksplisit."""

    def setUp(self) -> None:
        self.model = _MockBackboneModel().eval()
        self.gradcam = GradCAM(self.model, self.model.denseblock4)
        self.input_tensor = torch.randn(1, 3, 32, 32)

    def test_batched_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.gradcam.generate(torch.randn(4, 3, 32, 32), target_class=0)

    def test_invalid_target_class_raises(self) -> None:
        # Model satu-logit: hanya target_class=0 valid; dulu IndexError canggung.
        with self.assertRaises(ValueError) as ctx:
            self.gradcam.generate(self.input_tensor, target_class=1)
        self.assertIn("target_class", str(ctx.exception))

    def test_valid_generation_returns_normalized_heatmap(self) -> None:
        cam = self.gradcam.generate(self.input_tensor, target_class=0)
        self.assertEqual(cam.shape, (32, 32))
        self.assertGreaterEqual(cam.min(), 0.0)
        self.assertLessEqual(cam.max(), 1.0)

    def test_unused_target_layer_raises(self) -> None:
        unused = torch.nn.Conv2d(3, 4, 1)
        gradcam = GradCAM(self.model, unused)
        with self.assertRaises(RuntimeError):
            gradcam.generate(self.input_tensor, target_class=0)

    def test_empty_dataloader_raises_not_saves(self) -> None:
        # Dulu: figure kosong tersimpan + print "Grad-CAM saved".
        from torch.utils.data import DataLoader, TensorDataset

        empty = DataLoader(TensorDataset(torch.zeros(0, 3, 32, 32), torch.zeros(0)))
        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaises(RuntimeError):
                visualize_gradcam(
                    self.model,
                    empty,
                    torch.device("cpu"),
                    num_samples=2,
                    save_dir=out_dir,
                )
            self.assertFalse((Path(out_dir) / "gradcam_visualization.png").exists())


class HypothesesFailurePaths(unittest.TestCase):
    """Verdict H1-H3 harus menolak hasil yang tidak layak accepted."""

    @staticmethod
    def row(row_id: str, **metrics) -> dict:
        return {"id": row_id, **metrics}

    def test_tolerance_value_locked(self) -> None:
        from experiments.hypotheses import METRIC_TOLERANCE

        self.assertEqual(METRIC_TOLERANCE, 0.005)

    def test_missing_rows_are_skipped(self) -> None:
        from experiments.hypotheses import evaluate_hypotheses

        self.assertEqual(evaluate_hypotheses([]), {})

    def test_partial_run_only_evaluates_available_hypotheses(self) -> None:
        from experiments.hypotheses import evaluate_hypotheses

        rows = [
            self.row(
                "E4",
                accuracy=0.90,
                auc=0.93,
                total_params=11_416_961,
                infer_ms_cpu=50.0,
            ),
            self.row(
                "E6", accuracy=0.90, auc=0.93, total_params=7_347_683, infer_ms_cpu=40.0
            ),
        ]
        verdicts = evaluate_hypotheses(rows)
        self.assertNotIn("H1", verdicts)
        self.assertNotIn("H2", verdicts)
        self.assertIn("H3", verdicts)

    def test_h1_identical_results_not_accepted(self) -> None:
        from experiments.hypotheses import evaluate_hypotheses

        rows = [
            self.row("E1", accuracy=0.80, auc=0.90),
            self.row("E2", accuracy=0.80, auc=0.90),
        ]
        self.assertEqual(evaluate_hypotheses(rows)["H1"]["status"], "not_accepted")

    def test_h1_improvement_accepted(self) -> None:
        from experiments.hypotheses import evaluate_hypotheses

        rows = [
            self.row("E1", accuracy=0.80, auc=0.90),
            self.row("E2", accuracy=0.85, auc=0.91),
        ]
        self.assertEqual(evaluate_hypotheses(rows)["H1"]["status"], "accepted")

    def test_h1_regression_beyond_tolerance_not_accepted(self) -> None:
        # Akurasi naik tapi AUC turun 0.02 (> toleransi) -> bukan accepted.
        from experiments.hypotheses import evaluate_hypotheses

        rows = [
            self.row("E1", accuracy=0.80, auc=0.90),
            self.row("E2", accuracy=0.85, auc=0.88),
        ]
        self.assertEqual(evaluate_hypotheses(rows)["H1"]["status"], "not_accepted")

    def test_h1_regression_at_tolerance_boundary_accepted(self) -> None:
        # AUC turun tepat 0.005 -> masih dalam toleransi.
        from experiments.hypotheses import evaluate_hypotheses

        rows = [
            self.row("E1", accuracy=0.80, auc=0.90),
            self.row("E2", accuracy=0.85, auc=0.895),
        ]
        self.assertEqual(evaluate_hypotheses(rows)["H1"]["status"], "accepted")

    def test_h2_f1_regression_beyond_tolerance_not_accepted(self) -> None:
        from experiments.hypotheses import evaluate_hypotheses

        rows = [
            self.row("E3", f1=0.84, auc=0.88),
            self.row("E4", f1=0.80, auc=0.92),
            self.row("E5", f1=0.83, auc=0.90),
        ]
        self.assertEqual(evaluate_hypotheses(rows)["H2"]["status"], "not_accepted")

    def test_h2_auc_gain_with_stable_f1_accepted(self) -> None:
        from experiments.hypotheses import evaluate_hypotheses

        rows = [
            self.row("E3", f1=0.84, auc=0.88),
            self.row("E4", f1=0.84, auc=0.92),
        ]
        self.assertEqual(evaluate_hypotheses(rows)["H2"]["status"], "accepted")

    def test_h3_equal_params_not_accepted(self) -> None:
        # Param E6 tidak lebih sedikit -> klaim efisiensi H3 gugur.
        from experiments.hypotheses import evaluate_hypotheses

        rows = [
            self.row(
                "E4",
                accuracy=0.90,
                auc=0.93,
                total_params=11_416_961,
                infer_ms_cpu=50.0,
            ),
            self.row(
                "E6",
                accuracy=0.90,
                auc=0.93,
                total_params=11_416_961,
                infer_ms_cpu=40.0,
            ),
        ]
        self.assertEqual(evaluate_hypotheses(rows)["H3"]["status"], "not_accepted")

    def test_h3_worse_latency_not_accepted(self) -> None:
        from experiments.hypotheses import evaluate_hypotheses

        rows = [
            self.row(
                "E4",
                accuracy=0.90,
                auc=0.93,
                total_params=11_416_961,
                infer_ms_cpu=40.0,
            ),
            self.row(
                "E6", accuracy=0.90, auc=0.93, total_params=7_347_683, infer_ms_cpu=60.0
            ),
        ]
        self.assertEqual(evaluate_hypotheses(rows)["H3"]["status"], "not_accepted")

    def test_h3_equal_metrics_within_tolerance_accepted(self) -> None:
        rows = [
            self.row(
                "E4",
                accuracy=0.90,
                auc=0.93,
                total_params=11_416_961,
                infer_ms_cpu=50.0,
            ),
            self.row(
                "E6",
                accuracy=0.895,
                auc=0.93,
                total_params=7_347_683,
                infer_ms_cpu=40.0,
            ),
        ]
        from experiments.hypotheses import evaluate_hypotheses

        self.assertEqual(evaluate_hypotheses(rows)["H3"]["status"], "accepted")

    def test_h3_missing_latency_does_not_block(self) -> None:
        # infer_ms_cpu opsional: tak ada -> tidak menahan verdict.
        from experiments.hypotheses import evaluate_hypotheses

        rows = [
            self.row("E4", accuracy=0.90, auc=0.93, total_params=11_416_961),
            self.row("E6", accuracy=0.90, auc=0.93, total_params=7_347_683),
        ]
        self.assertEqual(evaluate_hypotheses(rows)["H3"]["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
