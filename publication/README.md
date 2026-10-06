# Publication

Elsevier CAS single-column manuscript for the PCOS classification study.

## Layout

```text
publication/
├── README.md               # this file
├── Makefile                # `make` builds manuscript/build/main.pdf
├── .gitignore              # LaTeX build artifacts
├── manuscript/             # YOUR WORK — edit only this directory
│   ├── main.tex            # manuscript (cas-sc, pre-filled H1–H3 / E1–E6 scaffold)
│   ├── references.bib      # real refs (Tiwari 2026, Sundari 2025)
│   ├── cas-sc.cls / cas-common.sty / cas-model2-names.bst
│   │                       # local vendor copies so manuscript/ compiles standalone
│   ├── thumbnails/         # vendor social/email icons required by cas-common.sty
│   │                       # (copied from template/, do not edit)
│   ├── figures/            # put pipeline.pdf, gradcam.pdf, overlays here
│   └── build/              # generated PDF + aux (gitignored)
└── template/               # referensi CAS Bundle v2.4 (single-column saja) — DO NOT EDIT
    ├── cas-sc-template.tex # template asli Elsevier (acuan main.tex)
    ├── cas-sc.cls / cas-common.sty / cas-model2-names.bst
    ├── thumbnails/         # ikon vendor
    └── README, manifest.txt
```

## Build

```bash
# from publication/
make          # -> manuscript/build/main.pdf
make clean    # remove build dir

# or manually:
cd manuscript
latexmk -pdf -outdir=build main.tex
```

Full sequence (`pdflatex → bibtex → pdflatex ×2`) is handled by `latexmk`
via `manuscript/.latexmkrc`. The `cas-sc.cls` / `cas-common.sty` / `.bst`
copies inside `manuscript/` are intentional: the directory compiles without
`TEXINPUTS` tricks and matches what Elsevier expects on submission
(upload `main.tex`, `references.bib`, `figures/*`, plus the `.cls/.sty/.bst`).

## Filling in results

| Placeholder in `main.tex` | Source |
|---|---|
| Table `tbl:results` (E1–E6 acc/F1/AUC/params/latency) | `experiments/results/results.csv` |
| Table `tbl:quality` (PSNR/SSIM, ROI stats) | `experiments/quality/summary.txt` |
| Fig `fig:pipeline` | export pipeline diagram to `manuscript/figures/pipeline.pdf` |
| Fig `fig:gradcam` | export Grad-CAM grids to `manuscript/figures/gradcam.pdf` |
| Abstract, Discussion, Conclusion `TODO` | narrative source: `references/kajian-literatur-gap-hipotesis.md` |

Scope reminders (already encoded in the scaffold): single 70/15/15 split,
so comparisons stay descriptive; segmentation is unsupervised (no Dice);
G1/G2/G3 are future work.
