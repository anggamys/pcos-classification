"""Unsupervised follicle-candidate segmentation for ultrasound images.

Uses only NumPy/SciPy/PIL (no OpenCV / scikit-image) so it runs in the
project venv and on Colab without extra installs.

Follicles appear as dark (hypoechoic) rounded regions, so the pipeline is:
grayscale -> threshold (Otsu or adaptive mean) -> invert to dark regions ->
morphological cleanup -> connected components -> size filtering.
"""

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi


def to_grayscale_array(pil_image):
    gray = ImageOps.grayscale(pil_image)
    return np.array(gray).astype(np.float64)


def otsu_threshold(gray, nbins=256):
    """Compute Otsu threshold for a grayscale array in [0, 255]."""
    hist, bin_edges = np.histogram(gray.ravel(), bins=nbins, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127.0
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist) / total
    weight2 = 1.0 - weight1
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(np.cumsum(hist), 1e-10)
    global_mean = (hist * bin_centers).sum() / total
    mean2 = (global_mean - weight1 * mean1) / np.maximum(weight2, 1e-10)
    between = weight1 * weight2 * (mean1 - mean2) ** 2
    between[~np.isfinite(between)] = 0.0
    return float(bin_centers[int(np.argmax(between))])


def adaptive_mean_threshold(gray, window=31, offset=7.0):
    """Local-mean adaptive threshold (dark pixels below local mean - offset)."""
    if window % 2 == 0:
        window += 1
    local_mean = ndi.uniform_filter(gray.astype(np.float64), size=window)
    return local_mean - offset


def _disk(radius):
    r = int(max(radius, 1))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def cleanup_mask(mask, open_radius=2, close_radius=3, min_area=80,
                 remove_border=True):
    cleaned = ndi.binary_opening(mask, structure=_disk(open_radius))
    cleaned = ndi.binary_closing(cleaned, structure=_disk(close_radius))
    labeled, _ = ndi.label(cleaned)
    if labeled.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    if remove_border:
        border_ids = set(np.unique(labeled[0, :])) | set(np.unique(labeled[-1, :]))
        border_ids |= set(np.unique(labeled[:, 0])) | set(np.unique(labeled[:, -1]))
        border_ids.discard(0)
        for bid in border_ids:
            labeled[labeled == bid] = 0
        labeled, _ = ndi.label(labeled > 0)
        if labeled.max() == 0:
            return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(np.ones_like(labeled), labeled, range(1, labeled.max() + 1))
    keep = {i + 1 for i, s in enumerate(sizes) if s >= min_area}
    return np.isin(labeled, list(keep)) if keep else np.zeros_like(mask, dtype=bool)


def segment_otsu(pil_image, open_radius=2, close_radius=3, min_area=80,
                 max_coverage=0.5, fallback_percentile=15.0):
    gray = to_grayscale_array(pil_image)
    thresh = otsu_threshold(gray)
    mask = cleanup_mask(gray < thresh, open_radius, close_radius, min_area)
    if mask.mean() > max_coverage:
        # Otsu latched onto background/tissue split; fall back to darkest percentile
        thresh = float(np.percentile(gray, fallback_percentile))
        mask = cleanup_mask(gray < thresh, open_radius, close_radius, min_area)
    return mask


def segment_adaptive(pil_image, window=31, offset=7.0,
                     open_radius=2, close_radius=3, min_area=80):
    gray = to_grayscale_array(pil_image)
    local_thresh = adaptive_mean_threshold(gray, window=window, offset=offset)
    dark = gray < local_thresh
    return cleanup_mask(dark, open_radius, close_radius, min_area)


def segment(pil_image, method="otsu", **kwargs):
    if method == "none":
        arr = to_grayscale_array(pil_image)
        return np.zeros(arr.shape, dtype=bool)
    if method == "otsu":
        return segment_otsu(pil_image, **kwargs)
    if method == "adaptive":
        return segment_adaptive(pil_image, **kwargs)
    raise ValueError(f"Unknown segmentation method: {method}")


def analyze_regions(mask):
    mask = np.asarray(mask, dtype=bool)
    labeled, num = ndi.label(mask)
    total = mask.size
    if num == 0:
        return {"num_regions": 0, "coverage": 0.0, "mean_area": 0.0,
                "max_area": 0.0, "bbox": None}
    sizes = ndi.sum(np.ones_like(labeled), labeled, range(1, num + 1))
    objects = ndi.find_objects(labeled)
    bboxes = []
    for i, sl in enumerate(objects):
        if sl is None:
            continue
        ys, xs = sl
        bboxes.append((xs.start, ys.start, xs.stop, ys.stop, float(sizes[i])))
    bboxes.sort(key=lambda b: b[4], reverse=True)
    return {
        "num_regions": int(num),
        "coverage": float(mask.sum() / total),
        "mean_area": float(np.mean(sizes)),
        "max_area": float(np.max(sizes)),
        "bbox": bboxes[0][:4] if bboxes else None,
    }


def union_bbox(mask, pad=8):
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, w)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, h)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def apply_roi_crop(pil_image, mask, pad=8):
    bbox = union_bbox(mask, pad=pad)
    if bbox is None:
        return pil_image
    return pil_image.crop(bbox)


def apply_masked(pil_image, mask, background=0):
    img = np.array(pil_image)
    m = np.asarray(mask, dtype=bool)
    if m.shape != img.shape[:2]:
        raise ValueError("Mask shape does not match image shape")
    if img.ndim == 3:
        m = m[:, :, None]
    return Image.fromarray(np.where(m, img, background).astype(np.uint8))


def overlay_mask(pil_image, mask, color=(255, 0, 0), alpha=90):
    base = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, color + (0,))
    alpha_layer = Image.fromarray((np.asarray(mask, dtype=bool) * alpha).astype(np.uint8))
    overlay.putalpha(alpha_layer)
    return Image.alpha_composite(base, overlay).convert("RGB")
