#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

canonical_root="results/model_0818/cond1/full_observed_fit_v1"
resume_root="$canonical_root/resume_after_interrupt"
workspace="$canonical_root/evaluation_after_8"
resume_config="configs/specific_models/model_0818_cond1_full_observed_fit_resume_108.yaml"
simulation_config="$workspace/model_evaluation_simulation_config.yaml"
simulation_dir="$workspace/simulation"
evaluation_dir="$workspace/model_evaluation"
subjects=(101 102 103 104 105 106 107 108)

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/tmp/model0818_after8_pycache

mkdir -p "$workspace"

if [[ ! -f "$resume_root/search/subject_108/best_hyperparams.json" ]]; then
    echo "[resume after 8] restarting interrupted subject 108 in a preserved, separate output directory"
    python -B scripts/run_model_0818_exploratory_observed_fit.py \
        --config "$resume_config" \
        --phase search-one \
        --subjects 108 \
        --parallel-budget 128
else
    echo "[resume after 8] subject 108 search cache_hit=true"
fi

for subject_id in 101 102 103 104 105 106 107; do
    test -f "$canonical_root/search/subject_${subject_id}/best_hyperparams.json"
done
test -f "$resume_root/search/subject_108/best_hyperparams.json"

if [[ ! -f "$simulation_config" ]]; then
    python -B scripts/prepare_model_0818_subset_evaluation.py \
        --subjects "${subjects[@]}" \
        --workspace "$workspace" \
        --additional-search-dir "$resume_root/search" \
        --filter-seed-count 16 \
        --hyper-base-seed 20260826
else
    echo "[resume after 8] evaluation config cache_hit=true"
fi

printf '%s\n' "${subjects[@]}" | xargs -P 8 -n 1 bash -c '
    subject_id="$1"
    target="results/model_0818/cond1/full_observed_fit_v1/evaluation_after_8/simulation/subjects/subject_${subject_id}.json"
    if [[ -f "$target" ]]; then
        echo "[resume after 8 simulation] subject=${subject_id} cache_hit=true"
        exit 0
    fi
    python -B -m src.Bayesian_state.run_simulation \
        --config "results/model_0818/cond1/full_observed_fit_v1/evaluation_after_8/model_evaluation_simulation_config.yaml" \
        --subjects "$subject_id"
' _

if [[ ! -f "$evaluation_dir/evaluation_manifest.json" ]]; then
    python -B -m src.Bayesian_state.run_model_evaluation \
        --input-dir "$simulation_dir" \
        --output-dir "$evaluation_dir" \
        --subjects "${subjects[@]}" \
        --eval-prediction-mode prior_t \
        --window-size 16 \
        --oral-mode center
else
    echo "[resume after 8] model evaluation cache_hit=true"
fi

echo "[resume after 8] standard evaluation complete"
