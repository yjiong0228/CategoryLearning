param(
    [string]$Python = "C:\Users\Ran\.conda\envs\cate_learn\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python -m src.Bayesian_state.run_hyper_then_simulation `
    --backend hyper_cd `
    --hyper-config configs/hyper_cd_cfg/pmh_cond1_hyper_cd_v10_test8.yaml `
    --execution-mode per-subject `
    --stage all `
    --skip-completed-hyper `
    --skip-completed-simulation `
    --generated-sim-config configs/simulation_cfg/generated_from_hyper/pmh_cond1_v10_test8_best.yaml `
    --sim-output-dir results/state-based-simulation/pmh/cond1_v10_test8 `
    --no-keep-logs
