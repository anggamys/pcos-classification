"""Unsupervised follicle-candidate segmentation for ultrasound images.

Uses only NumPy/SciPy/PIL (no OpenCV / scikit-image) so it runs in the
project venv and on Colab without extra installs.

Follicles appear as dark (hypoechoic) rounded regions, so the pipeline is:
grayscale -> threshold (Otsu or adaptive mean) -> invert to dark regions ->
morphological cleanup -> connected components -> size filtering.
"""

from typing import cast

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi


def to_grayscale_array(pil_image):
    """Konversi citra ke array grayscale float64 untuk komputasi ambang.

    Args:
        pil_image (PIL.Image.Image): Citra masukan.

    Returns:
        numpy.ndarray: Array 2D grayscale bertipe float64.
    """
    gray = ImageOps.grayscale(pil_image)
    return np.array(gray).astype(np.float64)


def otsu_threshold(gray, nbins=256):
    """Threshold Otsu: ambang yang memaksimalkan varians antar-kelas.

    Args:
        gray (numpy.ndarray): Array grayscale rentang [0, 255].
        nbins (int): Jumlah bin histogram (default 256).

    Returns:
        float: Nilai ambang Otsu.
    """
    hist, bin_edges = np.histogram(gray.ravel(), bins=nbins, range=(0, 255))
    hist = hist.astype(np.float64)
    pixel_total = hist.sum()
    if pixel_total == 0:
        return 127.0
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    background_weight = np.cumsum(hist) / pixel_total
    foreground_weight = 1.0 - background_weight
    background_mean = np.cumsum(hist * bin_centers) / np.maximum(np.cumsum(hist), 1e-10)
    global_mean = (hist * bin_centers).sum() / pixel_total
    foreground_mean = (global_mean - background_weight * background_mean) / np.maximum(
        foreground_weight, 1e-10
    )
    between_class_variance = (
        background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
    )
    between_class_variance[~np.isfinite(between_class_variance)] = 0.0
    return float(bin_centers[int(np.argmax(between_class_variance))])


def adaptive_mean_threshold(gray, window=31, offset=7.0):
    """Peta ambang adaptif: rata-rata lokal dikurangi offset per piksel.

    Piksel di bawah ambang lokalnya dianggap gelap (kandidat folikel),
    sehingga tahan terhadap iluminasi tak merata.

    Args:
        gray (numpy.ndarray): Array grayscale.
        window (int): Ukuran jendela rata-rata lokal (default 31).
        offset (float): Pengurang ambang (default 7.0).

    Returns:
        numpy.ndarray: Peta ambang seukuran citra masukan.
    """
    if window % 2 == 0:
        window += 1
    local_mean = ndi.uniform_filter(gray.astype(np.float64), size=window)
    return local_mean - offset


def _disk(radius):
    """Elemen struktur cakram untuk operasi morfologi biner.

    Args:
        radius (int): Jari-jari cakram dalam piksel.

    Returns:
        numpy.ndarray: Mask boolean berbentuk cakram.
    """
    disk_radius = int(max(radius, 1))
    row_coords, col_coords = np.ogrid[
        -disk_radius : disk_radius + 1, -disk_radius : disk_radius + 1
    ]
    return (
        col_coords * col_coords + row_coords * row_coords
    ) <= disk_radius * disk_radius


def _label_mask(mask):
    """Pelabelan komponen terhubung dengan anotasi tipe statis.

    Args:
        mask (numpy.ndarray): Mask boolean masukan.

    Returns:
        tuple: Pasangan (array berlabel, jumlah label).
    """
    return cast(tuple[np.ndarray, int], ndi.label(mask))


def cleanup_mask(mask, open_radius=2, close_radius=3, min_area=80, remove_border=True):
    """Bersihkan mask: morfologi, buang komponen tepi, saring luas minimum.

    Args:
        mask (numpy.ndarray): Mask boolean mentah.
        open_radius (int): Jari-jari opening untuk memutus jembatan noise.
        close_radius (int): Jari-jari closing untuk menutup lubang kecil.
        min_area (int): Luas minimum region dalam piksel.
        remove_border (bool): Buang komponen yang menyentuh tepi citra
            (umumnya background, bukan folikel).

    Returns:
        numpy.ndarray: Mask boolean hasil pembersihan.
    """
    cleaned = ndi.binary_opening(mask, structure=_disk(open_radius))
    cleaned = ndi.binary_closing(cleaned, structure=_disk(close_radius))
    labeled, num_labels = _label_mask(cleaned)
    if num_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    if remove_border:
        border_ids = set(np.unique(labeled[0, :]).tolist())
        border_ids.update(np.unique(labeled[-1, :]).tolist())
        border_ids.update(np.unique(labeled[:, 0]).tolist())
        border_ids.update(np.unique(labeled[:, -1]).tolist())
        border_ids.discard(0)
        for border_label_id in border_ids:
            labeled[labeled == border_label_id] = 0
        labeled, num_labels = _label_mask(labeled > 0)
        if num_labels == 0:
            return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(np.ones_like(labeled), labeled, range(1, num_labels + 1))
    keep = {
        region_index + 1
        for region_index, region_size in enumerate(sizes)
        if region_size >= min_area
    }
    return np.isin(labeled, list(keep)) if keep else np.zeros_like(mask, dtype=bool)


def segment_otsu(
    pil_image,
    open_radius=2,
    close_radius=3,
    min_area=80,
    max_coverage=0.5,
    fallback_percentile=15.0,
):
    """Segmentasi kandidat folikel dengan threshold Otsu + fallback persentil.

    Bila mask Otsu mencakup lebih dari `max_coverage` citra (Otsu mengunci
    split background/jaringan), dipakai ambang persentil tergelap.

    Args:
        pil_image (PIL.Image.Image): Citra masukan.
        open_radius (int): Jari-jari opening morfologi.
        close_radius (int): Jari-jari closing morfologi.
        min_area (int): Luas minimum region dalam piksel.
        max_coverage (float): Batas cakupan wajar sebelum fallback (default 0.5).
        fallback_percentile (float): Persentil gelap untuk ambang cadangan.

    Returns:
        numpy.ndarray: Mask boolean kandidat folikel.
    """
    gray = to_grayscale_array(pil_image)
    threshold = otsu_threshold(gray)
    mask = cleanup_mask(gray < threshold, open_radius, close_radius, min_area)
    if mask.mean() > max_coverage:
        # Otsu latched onto background/tissue split; fall back to darkest percentile
        threshold = float(np.percentile(gray, fallback_percentile))
        mask = cleanup_mask(gray < threshold, open_radius, close_radius, min_area)
    return mask


def segment_adaptive(
    pil_image, window=31, offset=7.0, open_radius=2, close_radius=3, min_area=80
):
    """Segmentasi kandidat folikel dengan threshold rata-rata lokal adaptif.

    Args:
        pil_image (PIL.Image.Image): Citra masukan.
        window (int): Ukuran jendela rata-rata lokal.
        offset (float): Pengurang ambang lokal.
        open_radius (int): Jari-jari opening morfologi.
        close_radius (int): Jari-jari closing morfologi.
        min_area (int): Luas minimum region dalam piksel.

    Returns:
        numpy.ndarray: Mask boolean kandidat folikel.
    """
    gray = to_grayscale_array(pil_image)
    local_threshold = adaptive_mean_threshold(gray, window=window, offset=offset)
    dark_pixels = gray < local_threshold
    return cleanup_mask(dark_pixels, open_radius, close_radius, min_area)


def segment(pil_image, method="otsu", **kwargs):
    """Dispatcher segmentasi kandidat folikel.

    Args:
        pil_image (PIL.Image.Image): Citra masukan.
        method (str): "none" (mask kosong), "otsu", atau "adaptive".
        **kwargs: Parameter lanjutan diteruskan ke fungsi metode terpilih.

    Returns:
        numpy.ndarray: Mask boolean kandidat folikel.

    Raises:
        ValueError: Jika nama metode tidak dikenal.
    """
    if method == "none":
        arr = to_grayscale_array(pil_image)
        return np.zeros(arr.shape, dtype=bool)
    if method == "otsu":
        return segment_otsu(pil_image, **kwargs)
    if method == "adaptive":
        return segment_adaptive(pil_image, **kwargs)
    raise ValueError(f"Unknown segmentation method: {method}")


def analyze_regions(mask):
    """Hitung statistik region pada mask (untuk evaluasi sisi PCD).

    Args:
        mask (numpy.ndarray): Mask boolean kandidat folikel.

    Returns:
        dict: Cacah region (num_regions), proporsi cakupan (coverage),
            luas rata-rata (mean_area), luas maksimum (max_area), dan kotak
            pembatas region terbesar (bounding_box, None bila kosong).
    """
    mask = np.asarray(mask, dtype=bool)
    labeled, region_count = _label_mask(mask)
    total_pixels = mask.size
    if region_count == 0:
        return {
            "num_regions": 0,
            "coverage": 0.0,
            "mean_area": 0.0,
            "max_area": 0.0,
            "bounding_box": None,
        }
    sizes = ndi.sum(np.ones_like(labeled), labeled, range(1, region_count + 1))
    region_slices = ndi.find_objects(labeled)
    bounding_boxes = []
    for region_index, region_slice in enumerate(region_slices):
        if region_slice is None:
            continue
        y_coords, x_coords = region_slice
        bounding_boxes.append(
            (
                x_coords.start,
                y_coords.start,
                x_coords.stop,
                y_coords.stop,
                float(sizes[region_index]),
            )
        )
    bounding_boxes.sort(key=lambda box: box[4], reverse=True)
    return {
        "num_regions": int(region_count),
        "coverage": float(mask.sum() / total_pixels),
        "mean_area": float(np.mean(sizes)),
        "max_area": float(np.max(sizes)),
        "bounding_box": bounding_boxes[0][:4] if bounding_boxes else None,
    }


def union_bounding_box(mask, pad=8):
    """Kotak pembatas gabungan seluruh region + padding.

    Args:
        mask (numpy.ndarray): Mask boolean kandidat folikel.
        pad (int): Padding di tiap sisi dalam piksel (default 8).

    Returns:
        tuple atau None: (x_min, y_min, x_max, y_max); None bila mask kosong
            atau kotak tidak valid.
    """
    y_coords, x_coords = np.nonzero(np.asarray(mask, dtype=bool))
    if len(x_coords) == 0:
        return None
    height, width = mask.shape
    x_min, x_max = (
        max(int(x_coords.min()) - pad, 0),
        min(int(x_coords.max()) + pad, width),
    )
    y_min, y_max = (
        max(int(y_coords.min()) - pad, 0),
        min(int(y_coords.max()) + pad, height),
    )
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)


def apply_roi_crop(pil_image, mask, pad=8):
    """Potong citra ke kotak pembatas gabungan region (mode input "roi").

    Args:
        pil_image (PIL.Image.Image): Citra masukan.
        mask (numpy.ndarray): Mask boolean kandidat folikel.
        pad (int): Padding di tiap sisi dalam piksel (default 8).

    Returns:
        PIL.Image.Image: Citra hasil crop; citra asli bila mask kosong.
    """
    bounding_box = union_bounding_box(mask, pad=pad)
    if bounding_box is None:
        return pil_image
    return pil_image.crop(bounding_box)


def apply_masked(pil_image, mask, background=0):
    """Pertahankan piksel region, nolkan background (mode input "masked").

    Args:
        pil_image (PIL.Image.Image): Citra masukan.
        mask (numpy.ndarray): Mask boolean kandidat folikel.
        background (int): Nilai piksel background (default 0).

    Returns:
        PIL.Image.Image: Citra dengan background diganti nilai background.

    Raises:
        ValueError: Jika ukuran mask tidak sama dengan ukuran citra.
    """
    image_array = np.array(pil_image)
    binary_mask = np.asarray(mask, dtype=bool)
    if binary_mask.shape != image_array.shape[:2]:
        raise ValueError("Mask shape does not match image shape")
    if image_array.ndim == 3:
        binary_mask = binary_mask[:, :, None]
    return Image.fromarray(
        np.where(binary_mask, image_array, background).astype(np.uint8)
    )


def overlay_mask(pil_image, mask, color=(255, 0, 0), alpha=90):
    """Tumpangkan mask transparan di atas citra untuk visualisasi laporan.

    Args:
        pil_image (PIL.Image.Image): Citra masukan.
        mask (numpy.ndarray): Mask boolean kandidat folikel.
        color (tuple): Warna overlay RGB (default merah).
        alpha (int): Opasitas overlay 0-255 (default 90).

    Returns:
        PIL.Image.Image: Citra RGB dengan overlay mask.
    """
    base = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, color + (0,))
    alpha_layer = Image.fromarray(
        (np.asarray(mask, dtype=bool) * alpha).astype(np.uint8)
    )
    overlay.putalpha(alpha_layer)
    return Image.alpha_composite(base, overlay).convert("RGB")
