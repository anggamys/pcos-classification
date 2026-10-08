import numpy as np
import pywt
from PIL import Image
from scipy.ndimage import gaussian_filter


def bayes_shrink_denoise(channel, wavelet="db4", level=3):
    """Denoising kanal tunggal dengan threshold adaptif BayesShrink.

    Noise diestimasi dari median koefisien detail terhalus, lalu tiap
    koefisien detail di-soft-threshold secara adaptif sebelum rekonstruksi.

    Args:
        channel (numpy.ndarray): Array 2D satu kanal citra, rentang [0, 1].
        wavelet (str): Nama wavelet diskret (default "db4").
        level (int): Level dekomposisi wavelet (default 3).

    Returns:
        numpy.ndarray: Array 2D hasil denoise, bentuk sama dengan masukan.
    """
    coeffs = pywt.wavedec2(channel, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745

    denoised_coeffs = [coeffs[0]]
    for detail_level in coeffs[1:]:
        denoised_detail = []
        for detail_coeff in detail_level:
            variance = np.mean(detail_coeff**2)
            sigma_x = max(np.sqrt(max(variance - sigma**2, 0)), 1e-10)
            threshold = sigma**2 / sigma_x
            denoised_coeff = pywt.threshold(detail_coeff, threshold, mode="soft")
            denoised_detail.append(denoised_coeff)
        denoised_coeffs.append(denoised_detail)

    return pywt.waverec2(denoised_coeffs, wavelet)


def wavelet_denoise(pil_image):
    """Denoising citra dengan BayesShrink per kanal warna.

    Args:
        pil_image (PIL.Image.Image): Citra masukan (grayscale atau RGB).

    Returns:
        PIL.Image.Image: Citra hasil denoise, ukuran sama dengan masukan.
    """
    img = np.array(pil_image).astype(np.float64) / 255.0

    if img.ndim == 3 and img.shape[2] == 3:
        denoised = np.stack(
            [
                np.clip(bayes_shrink_denoise(img[:, :, channel_index]), 0, 1)
                for channel_index in range(3)
            ],
            axis=-1,
        )
    else:
        denoised = np.clip(bayes_shrink_denoise(img), 0, 1)

    denoised = (denoised * 255).astype(np.uint8)
    return Image.fromarray(denoised)


def gaussian_smooth(pil_image, sigma=0.5):
    """Penghalusan Gaussian ringan untuk sisa noise setelah denoising.

    Args:
        pil_image (PIL.Image.Image): Citra masukan.
        sigma (float): Standar deviasi kernel Gaussian (default 0.5).

    Returns:
        PIL.Image.Image: Citra hasil penghalusan, ukuran sama dengan masukan.
    """
    img = np.array(pil_image).astype(np.float64)
    smoothed = gaussian_filter(img, sigma=sigma, axes=(0, 1))
    return Image.fromarray(smoothed.astype(np.uint8))
