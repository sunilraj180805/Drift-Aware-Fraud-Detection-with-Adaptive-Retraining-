# Drift-Aware Fraud Detection with Adaptive Retraining

A research pipeline studying how financial fraud-detection models degrade under real-world distribution shift, and whether **unsupervised drift signals** can reliably trigger retraining that recovers performance — without ever needing new labels.

## Problem

Fraud detection models trained once and deployed statically collapse when transaction patterns shift over time. On the [Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), a documented dark-market shutdown around time step ~43 (Weber et al., 2019) causes exactly this kind of drift — giving a rare ground-truth event to validate an unsupervised detector against.

**Research question:** Can heterogeneous unsupervised drift signals reliably identify temporal distribution changes and trigger selective model adaptation that recovers fraud-detection performance, while controlling unnecessary retraining?

## Approach

1. **Static baseline degradation analysis** — train once on an early, pre-drift window (steps 1–20) and evaluate the frozen model across the full 49-step horizon to characterize failure.
2. **Unsupervised drift detection** (no labels used):
   - Population Stability Index (PSI), calibrated from the dataset's own in-training-period null distribution
   - Maximum Mean Discrepancy (MMD) on PCA-reduced embeddings between consecutive windows
   - Page-Hinkley test on prediction confidence
3. **Drift validation** — correlate the unsupervised drift score against the actual (label-based) performance drop.
4. **Three adaptation strategies** compared across three model architectures:
   - Strategies: static (no adaptation), periodic/sliding-window retraining, drift-triggered retraining
   - Architectures: XGBoost (tabular), GraphSAGE + XGBoost (hybrid), TGAT + XGBoost (time-aware hybrid)
5. **Rigor checks**: temporal-leakage audits, PSI threshold calibration, reference-window studies (fixed/rolling/expanding), retraining cooldown analysis, multi-seed robustness testing (n=10), and paired statistical significance testing (t-test, Wilcoxon, Bonferroni correction).

## Key Results

| Strategy | Post-Drift F1 (mean ± std, 10 seeds) | Post-Drift AUCPR | Retraining Cost |
|---|---|---|---|
| Static (no adaptation) | 0.009 ± 0.002 | 0.079 ± 0.005 | None |
| Periodic (sliding window) | 0.224 ± 0.020 | 0.445 ± 0.019 | Medium–High |
| **Drift-triggered** | **0.390 ± 0.015** | **0.532 ± 0.006** | Medium |

- Drift-triggered retraining improved post-drift F1 by **~41x** over the static baseline — statistically significant (paired t-test, t=80.37, p < 0.0001; Wilcoxon confirmed).
- Unsupervised drift signals correlated strongly with the actual ground-truth performance collapse: **PSI r=0.555, MMD r=0.634** (both p < 0.0001) — meaning the detector would have caught the real event without ever seeing a label.
- Drift-triggered retraining also beat periodic retraining on F1-per-retrain efficiency across cooldown and budget-constrained variants.
- Results held consistently across all three architectures (XGBoost, GraphSAGE+XGBoost, TGAT+XGBoost).

## Repository Structure

```
.
├── notebooks/
│   └── elliptic_drift_adaptation.ipynb   # Full experimental pipeline (upload this)
├── results/                              # Generated on run: figures, tables, CSVs
│   ├── drift_detectors/
│   ├── retraining/
│   ├── statistics/
│   └── final/
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/sunilraj180805/drift-aware-fraud-detection.git
cd drift-aware-fraud-detection
pip install -r requirements.txt
```

Download the [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) from Kaggle and place the following files in `./elliptic_bitcoin_dataset/`:
- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

Then run the notebook top to bottom:
```bash
jupyter notebook notebooks/elliptic_drift_adaptation.ipynb
```

A GPU (CUDA) is optional but speeds up XGBoost training and is required for the GraphSAGE/TGAT extensions; the notebook falls back to CPU automatically if none is detected.

## Methodology Notes & Limitations

- **Non-standard split by design:** the notebook trains on an early pre-drift window and evaluates across the full horizon (rather than a chronological 70/15/15 split) to deliberately surface post-drift degradation for study.
- **Single documented drift event:** Elliptic has one labeled distribution-shift event (~step 43); sensitivity analysis is run across boundaries 40–45 to test robustness to the exact timing.
- **Simplified time-aware GAT:** the TGAT-style architecture approximates time-aware neighbor weighting using per-node time steps (Elliptic has no per-edge timestamps), so it is not a full TGAT implementation.
- **Detector false alarms:** the unsupervised detectors do produce some pre-drift false alarms; this is measured and reported rather than hidden.

All of the above is documented and audited in the notebook itself (see the "Experimental and Leakage Audit" section).

## Tech Stack

Python · XGBoost · PyTorch · PyTorch Geometric (GraphSAGE, GAT) · scikit-learn · River (ADWIN/DDM/EDDM) · pandas · NumPy · SciPy (statistical testing) · matplotlib/seaborn

## References

Weber, M., et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* KDD 2019 Workshop on Anomaly Detection in Finance.
