"""Image-quality metrics and simple enhancement (NumPy/PIL only)."""

import math

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter


def enhance_clahe(pil_image, tiles=8, clip=2.0):
    """Simplified tile-based contrast enhancement (CLAHE-like)."""
    gray = ImageOps.grayscale(pil_image)
    arr = np.array(gray).astype(np.float64)
    height, width = arr.shape
    out = np.zeros_like(arr)
    tile_height, tile_width = max(height // tiles, 1), max(width // tiles, 1)
    for tile_y in range(0, height, tile_height):
        for tile_x in range(0, width, tile_width):
            tile = arr[tile_y : tile_y + tile_height, tile_x : tile_x + tile_width]
            hist, _ = np.histogram(tile, bins=256, range=(0, 255))
            limit = max(clip * tile.size / 256.0, 1.0)
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + excess / 256.0
            cumulative_hist = hist.cumsum()
            cumulative_hist = (cumulative_hist - cumulative_hist.min()) / max(
                cumulative_hist.max() - cumulative_hist.min(), 1e-10
            )
            lookup_table = cumulative_hist * 255.0
            pixel_index = np.clip(tile.astype(int), 0, 255)
            out[tile_y : tile_y + tile_height, tile_x : tile_x + tile_width] = (
                lookup_table[pixel_index]
            )
    out = Image.fromarray(out.astype(np.uint8)).convert("RGB")
    return out


def enhance(pil_image, method="none"):
    if method == "none":
        return pil_image
    if method == "clahe":
        return enhance_clahe(pil_image)
    raise ValueError(f"Unknown enhance method: {method}")


def _to_gray_float(pil_image):
    return np.array(ImageOps.grayscale(pil_image)).astype(np.float64)


def psnr(img_a, img_b, max_val=255.0):
    image_a = _to_gray_float(img_a)
    image_b = _to_gray_float(img_b)
    if image_a.shape != image_b.shape:
        image_b = np.array(
            ImageOps.grayscale(img_b).resize(image_a.shape[::-1])
        ).astype(np.float64)
    mse = float(np.mean((image_a - image_b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(max_val**2 / mse)


def ssim(img_a, img_b, max_val=255.0, sigma=1.5):
    """Simplified single-scale SSIM with Gaussian weighting."""
    image_a = _to_gray_float(img_a)
    image_b = _to_gray_float(img_b)
    if image_a.shape != image_b.shape:
        image_b = np.array(
            ImageOps.grayscale(img_b).resize(image_a.shape[::-1])
        ).astype(np.float64)
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    mu_a = gaussian_filter(image_a, sigma)
    mu_b = gaussian_filter(image_b, sigma)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    var_a = gaussian_filter(image_a * image_a, sigma) - mu_a2
    var_b = gaussian_filter(image_b * image_b, sigma) - mu_b2
    cov_ab = gaussian_filter(image_a * image_b, sigma) - mu_ab
    numerator = (2 * mu_ab + c1) * (2 * cov_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-10)))


def summarize(values):
    arr = np.array(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=float,
    )
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
