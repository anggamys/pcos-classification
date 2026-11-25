"""Generate Figure 1 (end-to-end pipeline diagram) for the JUTIF manuscript.

Run from the project root:

    .venv/bin/python publication/figures/generate_fig1.py

Output: publication/figures/fig1_pipeline.png (300 dpi).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "fig1_pipeline.png"


def add_box(ax, x, y, width, height, title, body, face, edge):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=2.0,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)

    if body:
        ax.text(
            x + width / 2,
            y + height * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#1a1a1a",
        )
        ax.text(
            x + width / 2,
            y + height * 0.33,
            body,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#1a1a1a",
            linespacing=1.6,
        )
    else:
        ax.text(
            x + width / 2,
            y + height / 2,
            title,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#1a1a1a",
        )


def add_arrow(ax, x_start, x_end, y):
    ax.annotate(
        "",
        xy=(x_end, y),
        xytext=(x_start, y),
        arrowprops={"arrowstyle": "-|>", "color": "#333333", "linewidth": 2.2},
    )


def main():
    fig, ax = plt.subplots(figsize=(13.4, 5.0))
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 42)
    ax.axis("off")

    ax.text(
        53,
        38.8,
        "End-to-End PCOS Classification Pipeline with Preprocessing, "
        "Ovarian ROI Generation, and Attention",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color="#1a1a1a",
    )

    y = 13.0
    height = 12.0
    boxes = [
        (2.0, 14.0, "Ultrasound Image", "(224 × 224 px)", "#dbeafe", "#2563eb"),
        (
            20.0,
            19.0,
            "Preprocessing",
            "• BayesShrink Denoising\n• Optional CLAHE",
            "#e5e7eb",
            "#4b5563",
        ),
        (
            43.0,
            19.5,
            "Unsupervised Ovarian\nROI Mask",
            "(Dark-Region Otsu +\nMorphology)",
            "#ffedd5",
            "#ea580c",
        ),
        (
            66.5,
            17.0,
            "Input Representation",
            "• Full Image\n• ROI Crop\n• Masked Input",
            "#dcfce7",
            "#16a34a",
        ),
    ]

    for x, width, title, body, face, edge in boxes:
        add_box(ax, x, y, width, height, title, body, face, edge)

    add_arrow(ax, 16.2, 19.7, y + height / 2)
    add_arrow(ax, 39.2, 42.7, y + height / 2)
    add_arrow(ax, 62.7, 66.2, y + height / 2)
    add_arrow(ax, 83.7, 87.2, y + height / 2)

    add_box(
        ax,
        87.5,
        13.0,
        15.5,
        12.0,
        "DenseNet-121 +\nAttention Module",
        "(Self-Attn / CBAM)",
        "#f3e8ff",
        "#9333ea",
    )
    ax.text(
        95.25,
        8.2,
        "Binary Classification &\nGrad-CAM Heatmaps",
        ha="center",
        va="center",
        fontsize=10,
        color="#1a1a1a",
        linespacing=1.5,
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
