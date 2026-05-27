# FFT Run Clustering

This folder provides clustering analysis for run-level accuracy trajectories cached in `schema_version=3` result files (`raw_runs_ref`).
Clustering is done **within each subject** (`subject_id`) independently.

## Input

- `subjects/subject_*.json` under a result directory (for example `results/state-based-grid-result/pmh/cond1/subjects`)
- `cache/subject_*_raw_runs.gz` referenced by each subject json

Each run record is expected to include:

- `subject_id`, `condition`, `run_index`
- `mean_error`, `params`
- `metrics_by_mode` with trajectory key (default mode: `prior_t`)

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

`cluster_mean_trajectories.png` now overlays two highlighted trajectories:

- model best-fit trajectory (`best fit`, solid black) with its assigned cluster label
- subject true trajectory (`subject true`, dashed red) with its assigned cluster label

`cluster_assignments.csv` includes:

- `cluster_label` (same as `active_cluster_label`)
- `raw_cluster_label` (argmax over all components)
- `active_cluster_label` (argmax over active components only)
- `cluster_confidence_active` (max posterior over active components)
- `cluster_prob_json` (posterior vector on active components)
- `component_weight_of_label` (global mixture weight of selected active component)

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
- `--dp-active-weight-threshold`

FFT feature options:

- `--fft-lowfreq-weighting` / `--no-fft-lowfreq-weighting`
- `--fft-lowfreq-weight-power`

Length policy:

- trajectories must have identical length within each subject
- mixed lengths are treated as data quality errors and the script stops with explicit run indices
