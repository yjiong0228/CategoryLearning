#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

analysis_config="configs/specific_models/model_0818_exploratory_observed_fit.yaml"
result_root="results/model_0818/cond1/exploratory_observed_fit_pre_recovery_v2"
search_dir="$result_root/search"
simulation_config="$result_root/model_evaluation_simulation_config.yaml"
simulation_dir="$result_root/simulation"
evaluation_dir="$result_root/model_evaluation"
log_path="$result_root/standard_evaluation_pipeline.log"

exec > >(tee -a "$log_path") 2>&1

echo "[0818 evaluation] waiting for 32 completed subject searches"
while true; do
    completed="$({ find "$search_dir" -mindepth 2 -maxdepth 2 -name best_hyperparams.json -type f 2>/dev/null || true; } | wc -l)"
    echo "[0818 evaluation] completed_searches=$completed/32"
    if [[ "$completed" -eq 32 ]]; then
        break
    fi
    sleep 30
done

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python -B scripts/run_model_0818_exploratory_observed_fit.py \
    --config "$analysis_config" \
    --phase evaluation-config

seq 101 132 | xargs -P 32 -n 1 bash -c '
    subject_id="$1"
    target="results/model_0818/cond1/exploratory_observed_fit_pre_recovery_v2/simulation/subjects/subject_${subject_id}.json"
    if [[ -f "$target" ]]; then
        echo "[0818 evaluation simulation] subject=${subject_id} cache_hit=true"
        exit 0
    fi
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        python -B -m src.Bayesian_state.run_simulation \
        --config "results/model_0818/cond1/exploratory_observed_fit_pre_recovery_v2/model_evaluation_simulation_config.yaml" \
        --subjects "$subject_id"
' _

python -B -m src.Bayesian_state.run_model_evaluation \
    --input-dir "$simulation_dir" \
    --output-dir "$evaluation_dir" \
    --eval-prediction-mode prior_t \
    --window-size 16 \
    --oral-mode center

echo "[0818 evaluation] standard model evaluation complete"

# Preserve the already authorized high-precision fit score as a distinct step.
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python -B scripts/run_model_0818_exploratory_observed_fit.py \
    --config "$analysis_config" \
    --phase rescore \
    --n-jobs 128

echo "[0818 evaluation] R128 x 128-seed final rescore complete"
