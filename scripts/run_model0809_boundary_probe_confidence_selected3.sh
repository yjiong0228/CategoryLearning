#!/usr/bin/env bash
set -euo pipefail

model0809_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
model0809_python="${MODEL0809_PYTHON:-python}"
model0809_hyper_config="$model0809_repo/configs/hyper_cd_cfg/model0809_cond1_boundary_probe_confidence_selected3.yaml"
model0809_generated_config="$model0809_repo/configs/simulation_cfg/generated_from_hyper/model0809_selected3_boundary_probe_confidence_best.yaml"
model0809_result_root="$model0809_repo/results/model_dynamic_continuous/boundary_probe_vonfidence_v2"
model0809_hyper_dir="$model0809_result_root/hyper_cd"
model0809_sim_dir="$model0809_result_root/simulation"
model0809_eval_dir="$model0809_result_root/model_evaluation"
model0809_log_dir="$model0809_result_root/logs"
model0809_log="$model0809_log_dir/model0809_boundary_probe_confidence_selected3.log"
model0809_status="$model0809_log_dir/model0809_boundary_probe_confidence_selected3.status"
model0809_pid_file="$model0809_log_dir/model0809_boundary_probe_confidence_selected3.pid"
model0809_lock="/tmp/model0809_boundary_probe_confidence_selected3.lock"
model0809_subjects=(103 105 120)

# Three concurrent subjects * sixteen Hyper-CD workers per subject = at most
# 48 processes. BLAS stays single-threaded to prevent hidden oversubscription.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/tmp

if ! command -v "$model0809_python" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$model0809_python" >&2
  exit 2
fi

mkdir -p "$model0809_log_dir"
exec 9>"$model0809_lock"
if ! flock -n 9; then
  printf 'not_started duplicate_lock %s\n' "$(date --iso-8601=seconds)" > "$model0809_status"
  exit 1
fi

# Existing subject winners are treated as immutable results. A repeated probe
# should use a new result directory/name instead of silently overwriting them.
for model0809_sid in "${model0809_subjects[@]}"; do
  if [[ -e "$model0809_hyper_dir/subject_${model0809_sid}/best_hyperparams.json" ]]; then
    printf 'not_started existing_result subject=%s %s\n' \
      "$model0809_sid" "$(date --iso-8601=seconds)" > "$model0809_status"
    exit 2
  fi
done

printf '%s\n' "$$" > "$model0809_pid_file"
model0809_exit_code=0
model0809_record_exit() {
  model0809_exit_code=$?
  if [[ $model0809_exit_code -eq 0 ]]; then
    printf 'completed %s\n' "$(date --iso-8601=seconds)" > "$model0809_status"
  else
    printf 'failed exit_code=%d %s\n' \
      "$model0809_exit_code" "$(date --iso-8601=seconds)" > "$model0809_status"
  fi
}
trap model0809_record_exit EXIT

printf 'running %s\n' "$(date --iso-8601=seconds)" > "$model0809_status"
cd "$model0809_repo"

{
  echo "Model 0809 selected-three confidence/retention boundary probe"
  echo "Started: $(date --iso-8601=seconds)"
  echo "Repository: $model0809_repo"
  echo "Subjects: ${model0809_subjects[*]}"
  echo "Hyper-CD topology: 3 concurrent subjects x up to 16 processes"
  echo "Probe coordinates: memory(gamma,w0), m, readout power, base lapse"
} >> "$model0809_log"

model0809_hyper_pids=()
for model0809_sid in "${model0809_subjects[@]}"; do
  model0809_subject_log="$model0809_log_dir/hyper_subject_${model0809_sid}.log"
  "$model0809_python" -u -c '
import sys
from pathlib import Path

from src.Bayesian_state.run_hyper_then_simulation import build_hyper_selector

config_path = Path(sys.argv[1])
subject_id = int(sys.argv[2])
optimizer = build_hyper_selector("hyper_cd", config_path)
optimizer.run_subject(subject_id, stage="all")
' "$model0809_hyper_config" "$model0809_sid" >> "$model0809_subject_log" 2>&1 &
  model0809_hyper_pids+=("$!")
  printf 'Hyper-CD subject %s started with PID %s\n' \
    "$model0809_sid" "${model0809_hyper_pids[-1]}" >> "$model0809_log"
done

model0809_hyper_failed=0
for model0809_index in "${!model0809_hyper_pids[@]}"; do
  model0809_sid="${model0809_subjects[$model0809_index]}"
  if wait "${model0809_hyper_pids[$model0809_index]}"; then
    printf 'Hyper-CD subject %s completed: %s\n' \
      "$model0809_sid" "$(date --iso-8601=seconds)" >> "$model0809_log"
  else
    printf 'Hyper-CD subject %s failed: %s\n' \
      "$model0809_sid" "$(date --iso-8601=seconds)" >> "$model0809_log"
    model0809_hyper_failed=1
  fi
done
if [[ $model0809_hyper_failed -ne 0 ]]; then
  exit 1
fi

"$model0809_python" -u -m src.Bayesian_state.run_hyper_then_simulation \
  --backend hyper_cd \
  --hyper-config "$model0809_hyper_config" \
  --subjects "${model0809_subjects[@]}" \
  --stage all \
  --skip-completed-hyper \
  --skip-simulation \
  --generated-sim-config "$model0809_generated_config" \
  --sim-output-dir "$model0809_sim_dir" \
  --keep-logs >> "$model0809_log" 2>&1

# The frozen finalist simulation uses eight 512-particle repeats per subject.
model0809_sim_pids=()
for model0809_sid in "${model0809_subjects[@]}"; do
  model0809_subject_log="$model0809_log_dir/simulation_subject_${model0809_sid}.log"
  "$model0809_python" -u -m src.Bayesian_state.run_simulation \
    --config "$model0809_generated_config" \
    --subjects "$model0809_sid" >> "$model0809_subject_log" 2>&1 &
  model0809_sim_pids+=("$!")
done

model0809_sim_failed=0
for model0809_index in "${!model0809_sim_pids[@]}"; do
  model0809_sid="${model0809_subjects[$model0809_index]}"
  if wait "${model0809_sim_pids[$model0809_index]}"; then
    printf 'Simulation subject %s completed: %s\n' \
      "$model0809_sid" "$(date --iso-8601=seconds)" >> "$model0809_log"
  else
    printf 'Simulation subject %s failed: %s\n' \
      "$model0809_sid" "$(date --iso-8601=seconds)" >> "$model0809_log"
    model0809_sim_failed=1
  fi
done
if [[ $model0809_sim_failed -ne 0 ]]; then
  exit 1
fi

"$model0809_python" -u -m src.Bayesian_state.run_model_evaluation \
  --input-dir "$model0809_sim_dir" \
  --output-dir "$model0809_eval_dir" \
  --subjects "${model0809_subjects[@]}" \
  --eval-prediction-mode prior_t >> "$model0809_log" 2>&1

printf 'Completed: %s\n' "$(date --iso-8601=seconds)" >> "$model0809_log"
