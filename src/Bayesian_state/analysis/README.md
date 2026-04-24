# FFT Run Clustering

This folder provides clustering analysis for run-level accuracy trajectories cached in `schema_version=3` result files (`raw_runs_ref`).

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

Outputs are written to `<input-dir>/analysis` by default:

- `cluster_assignments.csv`
- `fft_features.npy`
- `embedding_pca_2d.csv`
- `cluster_scatter.png`
- `cluster_mean_trajectories.png`
- `cluster_representative_trajectories.png`
- `analysis_meta.json`

## Methods

Supported clustering methods:

- `kmeans`
- `agglomerative`
- `dbscan`
- `gmm`

