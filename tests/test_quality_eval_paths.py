"""Test jalur gagal evaluasi kualitas PCD: data tidak ditemukan harus jelas.

Sebelumnya sampling kosong lolos ke `rows[0]` dan berakhir `IndexError`.
Sekarang kegagalan harus menyebut path, ekstensi yang ditemukan, dan langkah
perbaikannya.

Jalankan dari root repo:

    python -m unittest tests.test_quality_eval_paths -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import evaluate_quality


class SampleImagesFailureTests(unittest.TestCase):
    """Sampling tanpa citra .jpg harus gagal dengan pesan diagnostik."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_empty_data_dir_names_both_classes(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            evaluate_quality.sample_images(str(self.root), 5)

        message = str(ctx.exception)
        self.assertIn("infected", message)
        self.assertIn("noninfected", message)
        self.assertIn("--data-dir", message)

    def test_missing_class_dir_is_reported(self) -> None:
        (self.root / "infected").mkdir()
        (self.root / "infected" / "a.jpg").write_bytes(b"a")

        with self.assertRaises(ValueError) as ctx:
            evaluate_quality.sample_images(str(self.root), 5)

        self.assertIn("noninfected/", str(ctx.exception))
        self.assertIn("tidak ditemukan", str(ctx.exception))

    def test_wrong_extension_names_found_extensions(self) -> None:
        # .JPG huruf besar tidak terbaca glob *.jpg; pesan harus menyebut
        # ekstensi yang benar-benar ada agar penyebabnya kelihatan.
        for cls in ("infected", "noninfected"):
            (self.root / cls).mkdir(parents=True)
            (self.root / cls / "a.PNG").write_bytes(cls.encode())

        with self.assertRaises(ValueError) as ctx:
            evaluate_quality.sample_images(str(self.root), 5)

        self.assertIn(".png", str(ctx.exception))
        self.assertIn("case-sensitive", str(ctx.exception))

    def test_zero_samples_per_class_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            evaluate_quality.sample_images(str(self.root), 0)

        self.assertIn(">= 1", str(ctx.exception))


class SampleImagesSuccessTests(unittest.TestCase):
    """Sampling normal tetap mengembalikan path per kelas."""

    def test_unique_images_are_sampled_and_deduped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            for cls in ("infected", "noninfected"):
                (root / cls).mkdir(parents=True)
                (root / cls / "a.jpg").write_bytes(cls.encode())
                (root / cls / "b.jpg").write_bytes(f"{cls}-b".encode())
                # Salinan byte-identik dari a.jpg harus dibuang sebelum sampling.
                (root / cls / "copy.jpg").write_bytes(cls.encode())

            picked = evaluate_quality.sample_images(str(root), 10)

        for cls in ("infected", "noninfected"):
            self.assertEqual(len(picked[cls]), 2)
            self.assertEqual({p.name for p in picked[cls]}, {"a.jpg", "b.jpg"})


if __name__ == "__main__":
    unittest.main()
