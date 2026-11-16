"""Test segmentasi pada skala kerja tetap dan fallback mask kosong.

Jalankan dari root repo:

    python -m unittest tests.test_segmentation_scale -v
"""

import unittest

import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Resampling

from src.preprocessing.segmentation import (
    SEGMENT_WORK_SIZE,
    apply_masked,
    apply_roi_crop,
    segment,
    segment_at_scale,
)


def make_image(width: int, height: int) -> Image.Image:
    """Citra sintetis: latar terang dengan lima region gelap bulat.

    Args:
        width (int): Lebar citra.
        height (int): Tinggi citra.

    Returns:
        PIL.Image.Image: Citra RGB sintetis.
    """
    array = np.full((height, width, 3), 220, dtype=np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    centers = [(0.2, 0.3), (0.5, 0.25), (0.75, 0.35), (0.35, 0.7), (0.65, 0.65)]

    for fx, fy in centers:
        cx, cy = int(fx * width), int(fy * height)
        radius = max(6, int(0.07 * min(width, height)))
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(35, 35, 35),
        )

    return image


class SegmentScaleTests(unittest.TestCase):
    def test_segment_returns_mask_matching_native_size(self) -> None:
        image = make_image(800, 600)
        mask = segment(image, method="otsu")

        self.assertEqual(mask.shape, (600, 800))
        self.assertEqual(mask.dtype, np.bool_)

    def test_segment_at_scale_returns_square_work_mask(self) -> None:
        image = make_image(800, 600)
        mask = segment_at_scale(image, method="otsu")

        self.assertEqual(mask.shape, (SEGMENT_WORK_SIZE, SEGMENT_WORK_SIZE))

    def test_coverage_is_comparable_across_input_resolutions(self) -> None:
        large = make_image(800, 600)
        small = large.resize((160, 120), Resampling.BILINEAR)

        coverage_large = segment(large, method="otsu").mean()
        coverage_small = segment(small, method="otsu").mean()

        self.assertGreater(coverage_large, 0.0)
        self.assertAlmostEqual(coverage_large, coverage_small, delta=0.05)

    def test_none_method_returns_native_empty_mask(self) -> None:
        mask = segment(make_image(300, 200), method="none")

        self.assertEqual(mask.shape, (200, 300))
        self.assertFalse(mask.any())

    def test_unknown_method_raises_through_scale_wrapper(self) -> None:
        with self.assertRaises(ValueError):
            segment_at_scale(make_image(320, 240), method="kmeans")


class EmptyMaskFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_image(320, 240)
        self.empty = np.zeros((240, 320), dtype=bool)
        self.region = np.zeros((240, 320), dtype=bool)
        self.region[50:150, 60:180] = True

    def test_masked_with_empty_mask_returns_original(self) -> None:
        result = apply_masked(self.image, self.empty)

        self.assertIs(result, self.image)
        np.testing.assert_array_equal(np.array(result), np.array(self.image))

    def test_masked_with_region_zeros_background(self) -> None:
        result = np.array(apply_masked(self.image, self.region))
        original = np.array(self.image)

        np.testing.assert_array_equal(result[50:150, 60:180], original[50:150, 60:180])
        self.assertTrue((result[0, 0] == 0).all())

    def test_roi_with_empty_mask_returns_original(self) -> None:
        result = apply_roi_crop(self.image, self.empty)

        self.assertIs(result, self.image)

    def test_masked_rejects_shape_mismatch(self) -> None:
        wrong = np.zeros((100, 100), dtype=bool)

        with self.assertRaises(ValueError):
            apply_masked(self.image, wrong)


if __name__ == "__main__":
    unittest.main()
