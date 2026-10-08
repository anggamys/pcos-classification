"""Command-line interface: argument definitions and config building."""

import argparse


def parse_args() -> argparse.Namespace:
    """Definisikan dan uraikan argumen baris perintah training.

    Returns:
        argparse.Namespace: Seluruh argumen CLI (data, preprocessing,
            model, training, Grad-CAM).
    """
    parser = argparse.ArgumentParser(
        description="Pengaruh Wavelet Denoising dan Segmentasi Folikel "
        "terhadap Klasifikasi PCOS dengan DenseNet-121 + Attention"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="Path to dataset directory (default: datasets)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image resize size (default: 224)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Max training epochs (default: 30)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--denoise", action="store_true", help="Enable wavelet denoising preprocessing"
    )
    parser.add_argument(
        "--enhance",
        type=str,
        default="none",
        choices=["none", "clahe"],
        help="Contrast enhancement (default: none)",
    )
    parser.add_argument(
        "--segment",
        type=str,
        default="none",
        choices=["none", "otsu", "adaptive"],
        help="Follicle segmentation method (default: none)",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="full",
        choices=["full", "roi", "masked"],
        help="Model input: full image, ROI crop, or masked (default: full)",
    )
    parser.add_argument(
        "--seg-pad",
        type=int,
        default=8,
        help="Padding around ROI crop in pixels (default: 8)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader num_workers (default: 2)"
    )

    parser.add_argument(
        "--attention",
        type=str,
        default="self_attention",
        choices=["self_attention", "se_net", "cbam", "transformer"],
        help="Attention mechanism (default: self_attention)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)"
    )
    parser.add_argument(
        "--gradcam", action="store_true", help="Generate Grad-CAM visualization"
    )
    parser.add_argument(
        "--gradcam-samples",
        type=int,
        default=5,
        help="Number of Grad-CAM samples (default: 5)",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Susun dict config training dari argumen CLI.

    Args:
        args (argparse.Namespace): Hasil `parse_args`.

    Returns:
        dict: Config (data_dir, batch_size, ..., seg_pad) untuk dataset
            dan training.
    """
    return {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "save_dir": args.save_dir,
        "denoise": args.denoise,
        "enhance": args.enhance,
        "segment": args.segment,
        "input_mode": args.input_mode,
        "seg_pad": args.seg_pad,
    }
