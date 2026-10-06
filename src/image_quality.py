"""Image-quality metrics and simple enhancement (NumPy/PIL only)."""

import math

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter


def enhance_clahe(pil_image, tiles=8, clip=2.0):
    """Simplified tile-based contrast enhancement (CLAHE-like)."""
    gray = ImageOps.grayscale(pil_image)
    arr = np.array(gray).astype(np.float64)
    h, w = arr.shape
    out = np.zeros_like(arr)
    th, tw = max(h // tiles, 1), max(w // tiles, 1)
    for y0 in range(0, h, th):
        for x0 in range(0, w, tw):
            tile = arr[y0:y0 + th, x0:x0 + tw]
            hist, _ = np.histogram(tile, bins=256, range=(0, 255))
            limit = max(clip * tile.size / 256.0, 1.0)
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + excess / 256.0
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) / max(cdf.max() - cdf.min(), 1e-10)
            lut = (cdf * 255.0)
            idx = np.clip(tile.astype(int), 0, 255)
            out[y0:y0 + th, x0:x0 + tw] = lut[idx]
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
    a = _to_gray_float(img_a)
    b = _to_gray_float(img_b)
    if a.shape != b.shape:
        b = np.array(ImageOps.grayscale(img_b).resize(a.shape[::-1])).astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def ssim(img_a, img_b, max_val=255.0, sigma=1.5):
    """Simplified single-scale SSIM with Gaussian weighting."""
    a = _to_gray_float(img_a)
    b = _to_gray_float(img_b)
    if a.shape != b.shape:
        b = np.array(ImageOps.grayscale(img_b).resize(a.shape[::-1])).astype(np.float64)
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    var_a = gaussian_filter(a * a, sigma) - mu_a2
    var_b = gaussian_filter(b * b, sigma) - mu_b2
    cov_ab = gaussian_filter(a * b, sigma) - mu_ab
    num = (2 * mu_ab + c1) * (2 * cov_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / np.maximum(den, 1e-10)))


def summarize(values):
    arr = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}
