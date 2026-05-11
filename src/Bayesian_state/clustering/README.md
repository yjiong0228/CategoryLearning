# FFT Run Clustering

This folder provides clustering analysis for run-level accuracy trajectories cached in `schema_version=3` result files (`raw_runs_ref`).
Clustering is done **within each subject** (`subject_id`) independently.

## Input

- `subject_*.json` under a result directory (for example `results/state-based-grid-result/pmh/cond1`)
- `cache/subject_*_raw_runs.gz` referenced by each subject json

Each run record is expected to include:

- `subject_id`, `condition`, `run_index`
- `mean_error`, `params`
- `metrics` with trajectory key (default: `sliding_pred_acc`)

## Usage

Run in the `cate_learn` conda environment:

```bash
conda run -n cate_learn python -m src.Bayesian_state.analysis.run_fft_clustering \
  --input-dir results/state-based-grid-result/pmh/cond1 \
  --method kmeans \
  --n-clusters 4 \
  --fft-keep-ratio 0.2
```

## Outputs

Outputs are written to `<input-dir>/analysis` by default.
Each subject gets an independent subfolder:

- `subject_<id>/cluster_assignments.csv`
- `subject_<id>/fft_features.npy`
- `subject_<id>/embedding_pca_2d.csv`
- `subject_<id>/cluster_scatter.png`
- `subject_<id>/cluster_mean_trajectories.png`
- `subject_<id>/cluster_representative_trajectories.png`
- `subject_<id>/clustering_report.json`

`cluster_assignments.csv` includes:

- `cluster_label`
- `cluster_confidence` (max posterior probability for that sample)
- `cluster_prob_json` (full posterior probability vector)

Top-level summary files:

- `cluster_assignments_all_subjects.csv`
- `embedding_pca_2d_all_subjects.csv`
- `analysis_meta.json`

## Methods

Supported clustering methods:

- `kmeans`
- `agglomerative`
- `dbscan`
- `gmm`
- `dpmm` (truncated DPMM via `BayesianGaussianMixture`)

For `dpmm`, additional CLI options are available:

- `--dp-max-components`
- `--dp-weight-concentration-prior`
- `--dp-covariance-type {full,diag}`
- `--dp-max-iter`
- `--dp-n-init`
