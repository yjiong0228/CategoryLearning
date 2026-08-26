#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

analysis_config="configs/specific_models/model_0818_cond1_full_observed_fit.yaml"
result_root="results/model_0818/cond1/full_observed_fit_v1"
simulation_config="$result_root/model_evaluation_simulation_config.yaml"
simulation_dir="$result_root/simulation"
evaluation_dir="$result_root/model_evaluation"
log_path="$result_root/full_standard_evaluation_pipeline.log"

mkdir -p "$result_root"
exec > >(tee -a "$log_path") 2>&1

echo "[0818 full fit] validating all-trial input scope"
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python -B scripts/run_model_0818_exploratory_observed_fit.py \
    --config "$analysis_config" \
    --phase validate

echo "[0818 full fit] searching 32 subjects with four concurrent 32-core jobs"
seq 101 132 | xargs -P 4 -n 1 bash -c '
    subject_id="$1"
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        python -B scripts/run_model_0818_exploratory_observed_fit.py \
        --config "configs/specific_models/model_0818_cond1_full_observed_fit.yaml" \
        --phase search-one \
        --subjects "$subject_id" \
        --parallel-budget 32
' _

completed="$({ find "$result_root/search" -mindepth 2 -maxdepth 2 -name best_hyperparams.json -type f 2>/dev/null || true; } | wc -l)"
if [[ "$completed" -ne 32 ]]; then
    echo "[0818 full fit] expected 32 completed searches, found $completed" >&2
    exit 1
fi

python -B scripts/run_model_0818_exploratory_observed_fit.py \
    --config "$analysis_config" \
    --phase evaluation-config

echo "[0818 full fit] generating standard logged simulations"
seq 101 132 | xargs -P 32 -n 1 bash -c '
    subject_id="$1"
    target="results/model_0818/cond1/full_observed_fit_v1/simulation/subjects/subject_${subject_id}.json"
    if [[ -f "$target" ]]; then
        echo "[0818 full evaluation simulation] subject=${subject_id} cache_hit=true"
        exit 0
    fi
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        python -B -m src.Bayesian_state.run_simulation \
        --config "results/model_0818/cond1/full_observed_fit_v1/model_evaluation_simulation_config.yaml" \
        --subjects "$subject_id"
' _

python -B -m src.Bayesian_state.run_model_evaluation \
    --input-dir "$simulation_dir" \
    --output-dir "$evaluation_dir" \
    --eval-prediction-mode prior_t \
    --window-size 16 \
    --oral-mode center

echo "[0818 full fit] standard model evaluation complete"

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python -B scripts/run_model_0818_exploratory_observed_fit.py \
    --config "$analysis_config" \
    --phase rescore \
    --n-jobs 128

echo "[0818 full fit] R128 x 128-seed final rescore complete"
