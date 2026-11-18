"""Test BayesShrink: guard subband tanpa noise dan keluaran finite.

Jalankan dari root repo:

    python -m unittest tests.test_wavelet -v
"""

import unittest

import numpy as np
from PIL import Image, ImageDraw

from src.preprocessing.wavelet import bayes_shrink_denoise, wavelet_denoise


class BayesShrinkGuardTests(unittest.TestCase):
    def test_flat_channel_is_returned_unchanged(self) -> None:
        channel = np.zeros((64, 64), dtype=np.float64)

        result = bayes_shrink_denoise(channel)

        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, channel)

    def test_flat_background_image_has_no_nan_pixels(self) -> None:
        array = np.zeros((96, 120, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
        draw.ellipse([30, 20, 80, 70], fill=(180, 180, 180))

        result = np.array(wavelet_denoise(image))

        self.assertEqual(result.shape, array.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse((result == 0).all())

    def test_noisy_channel_stays_finite_and_same_shape(self) -> None:
        rng = np.random.default_rng(42)
        channel = np.clip(rng.normal(0.5, 0.1, (80, 70)), 0, 1)

        result = bayes_shrink_denoise(channel)

        self.assertEqual(result.shape, channel.shape)
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
