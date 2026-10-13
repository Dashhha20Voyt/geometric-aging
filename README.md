# Beyond the Scalar Clock — ASCA-2026 code package

Geometric framework for assessing biological aging from DNA methylation
(GSE40279, whole blood, n = 656, ages 19–101).

Authors: Darya Voitenko, Dmitrii Kriukov, Ekaterina Khrameeva (Skoltech).
Funded by the Russian Science Foundation, grant 25-71-20017.

## Pipeline order

| # | Script | What it does | Key outputs |
|---|--------|--------------|-------------|
| 0 | `00_synthetic_validation.py` | Validates all four invariants (TDI, MV, BRS, TE) on synthetic manifolds with analytic ground truth: geodesics on S², saddle of a quasi-potential, Lyapunov/entropy rate, persistence-diagram stability. | `validation_results.json`, `validation_figure.png` |
| — | `utils_build_matrix.py`, `utils_extract_vocab_cpgs.py` | Stream the 1.2 GB GSE40279 beta matrix; build the 656-sample matrix and the MethylGPT-vocabulary-aligned matrix (45,309 fully observed CpGs). | `betas_sub.npz`, `methylgpt_input.npz` |
| 1 | `01_raw_beta_pipeline.py` | Baseline: invariants on raw beta values (top-2000 variable CpGs). | `real_results.json`, `real_figure.png` |
| 2 | `02_methylgpt_embeddings.py` | Extracts 64-d MethylGPT foundation-model embeddings (CPU; 2,000-CpG context; flash-attn → PyTorch weight remap). | `methylgpt_embeddings.npz` |
| 3 | `03_invariants_on_embeddings.py` | Invariants on MethylGPT embeddings. | `emb_results.json`, `emb_figure.png` |
| 4 | `04_cellcomp_phate_pipeline.py` | **Main pipeline.** Reference-free cell-composition correction (ReFACTor-style; nuisance PCs orthogonalised against age so the age axis is untouched) → PHATE diffusion embedding. | `phate_emb.npy`, `phate_Xadj.npy` |
| 5 | `05_invariants_downstream.py` | Geodesics, reference trajectory γ_H (Fréchet mean per 3-yr bin), TDI / MV / BRS / TE, 44/60 peak tests, age prediction (geodesic kNN vs ElasticNet). | `phate_results.json`, `phate_figure.png` |
| 6 | `06_make_figures.py` | Publication figures for the talk, incl. leave-one-out geodesic-kNN age estimation (MAE 3.9 y, r 0.94). | `figs/*.png` |

## Verified headline numbers (main pipeline)

- PHATE axis 1 vs chronological age: **r = 0.73** (unsupervised; no age labels used to build the map)
- Geodesic kNN age estimation: **held-out 70/30 MAE = 4.36 y (r = 0.93)**; **leave-one-out MAE = 3.89 y (r = 0.94)**
- MV peaks at 47.5 and 59.5 y; BRS peak at 41.5 y (3-yr bins — peaks localised to ±3 y)
- ElasticNet on the same matrix (supervised reference): MAE 1.29 y (r = 0.99)

Caveat: the embedding is transductive (built on the full cohort); only the age
read-out is cross-validated. No regressor is ever trained on age.

## Data

- GEO GSE40279 (Hannum et al., Mol Cell 2013): `GSE40279_average_beta.txt.gz`
  from https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/suppl/
- MethylGPT: https://github.com/albert-ying/MethylGPT (bundled base checkpoint)

## Environment

Python ≥ 3.10; numpy, scipy, scikit-learn, matplotlib, phate.
MethylGPT extraction additionally needs torch 2.2 (CPU) + torchtext in a
separate Python 3.12 venv.
