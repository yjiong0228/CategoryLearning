> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Local Cond1/2/3 Batch Preparation

## Complete-records Rerun (2026-11-04)

The increased-budget eight-worker configs are separate from the old twelve-worker
configs: `configs/local_records8/cond{1,2,3}_full_local.yaml`. They retain all 96
subjects and every available trial (trial counts vary by subject; they are not
fixed at 320/640). Their budgets are 64 starts, two rounds, 16 proposals per round;
search 32 particles x 2 seeds, rescore 128 x 4, audit 384 x 8; evaluation 512
particles x 16 PF repeats and 512 autonomous paths. Complete conditional particle
export is enabled. These budgets increase sampling over the previous local batch,
but are smaller than the proposed 64-worker server budgets.

```powershell
& './tools/run_cond123_local.ps1' -Jobs 8 -ConfigDirectory configs/local_records8 `
  -ExportPaper -PythonExe 'python' `
  -OutputRoot results/CHOOSE_NEW_DIRECTORY
```

`-ConfigDirectory` selects the config set without editing old experiments.
`-ExportPaper` adds a fourth sequential stage exporting rule/feature CSV and PNG
to each subject's `paper/` directory. The paper-export window is 16 in this
runner, matching this config set. Default invocation without either option is
unchanged. The plan records both options; `-PlanOnly` still creates no output.

Full-length S314 (832 rows, fitted capacity 8) completed all four stages in
127.4 seconds under eight workers with an eight-CPU affinity mask. Its conditional
archive has 512 particles, pre-feedback content state and all 152 rule columns;
the approximately 920 MiB archive passed checksum and probability checks.
This is a runtime/integration check, not evidence of fit quality or a guarantee
that every other subject will take the same time. The complete Python regression
suite passed 272 tests (two skips, 25 subtests) before batch launch.

The supervised background launch additionally uses `OMP_THREAD_LIMIT=8`, an
eight-logical-CPU affinity mask, a below-normal process priority and a four-hour
wall-time guard. On timeout only its own worker process tree is stopped; partial
outputs remain. Keep the computer awake. Existing results and remote machines
are not modified.

`tools/run_cond123_local.ps1` runs native C++ fitting, native evaluation, then
offline Python plotting for each selected subject. Conditions and subjects run
sequentially. Each native command uses one task-parallel layer with up to 12
workers by default. Numeric-library thread counts are set to one during the run
and restored afterward. No build, dependency installation, or remote operation
is performed by the script.

The three `configs/cond{1,2,3}_full_local.yaml` files each select the condition's
32 subjects from `data/exp123/processed/Task2_processed.csv`. They contain no
`max_trials` limit. All three preserve the complete fit budget and shared
parameter support used by `configs/cond1_full.yaml`: 48 initial starts, two
rounds, 16 proposals per round, and the existing search/rescore/audit settings.
Evaluation uses 256 particles, 12 independent particle-filter repeats, 128
autonomous simulations, 256 band draws, and a 16-trial rolling window.
Particle-filter repeats represent repeated stochastic inference; autonomous
simulations generate separate learning paths under task feedback.

Preparing these files does not start or validate the full 96-subject batch.
Full fitting requires an explicit decision to run it. The native Release binary
supports up to 12 workers on Windows. Python must provide the
dependencies declared in `evaluation/requirements.txt`. Before creating any
run output or starting a fit, the script checks that the selected interpreter
can import NumPy, SciPy, and Matplotlib from the same working directory and with
the same numeric-library thread bounds used for plotting. A failed import stops
the run and leaves
the requested output directory absent. Plotting reads saved C++ outputs only and
generates PNG figures together with numerical summaries and arrays.

## Invocation

Run the examples below from the repository root in an existing Windows PowerShell
5.1 or PowerShell 7 session. An absolute script path also works from any current
directory, including paths with spaces. Relative executable and output paths are
resolved against this repository,
not the caller's directory. A bare `-PythonExe python` resolves through `PATH`;
use an absolute interpreter path when multiple Python installations exist.

Inspect the full plan without running commands or creating an output directory:

```powershell
& './tools/run_cond123_local.ps1' -PlanOnly
```

The following command launches the full batch when explicitly authorized:

```powershell
& './tools/run_cond123_local.ps1' -Jobs 12 -PythonExe python -Executable 'build/Release/genmodel_cpp.exe'
```

The default output is a new timestamped `results/local_cond123_...` directory.
Supply `-OutputRoot 'results/my_new_run'` to choose a fresh destination. Existing
directories, including empty ones, are refused. `-PlanOnly` also checks this
constraint, but needs neither executable to exist and performs no model work.
Its standard output is JSON containing the ordered commands and argument arrays.
The dependency import probe runs only in execution mode.

The runner reads each config's subject selection in a restricted format: one
top-level `subjects: [101, 102, ...]` line containing decimal integer IDs separated
by commas, with optional spaces and no inline comment. Keep this single-line
flow-list format when editing the local configs. This makes dry planning
independent of Python and PyYAML; the native CLI parses the full YAML on execution.

## Bounded Smoke Check

A bounded real fit/evaluate/plot check uses four trials:

```powershell
& './tools/run_cond123_local.ps1' -Jobs 2 -Conditions 1 -Subjects 122 -MaxTrials 4 -Smoke
```

`-Smoke` passes the native `--smoke` flag to both fitting and evaluation. The
native CLI reduces search/evaluation budgets and caps the sequence at 16 trials;
`-MaxTrials 4` further restricts this example. `-MaxTrials` alone truncates the
sequence while retaining the full configured search budget. Small validation
runs should use both switches and at most two workers. These checks establish
software integration, not parameter recovery or predictive validity.

`-Conditions` accepts a unique subset of `1,2,3`. `-Subjects` accepts explicit
unique IDs and filters the selected conditions; an ID outside those conditions
is rejected. PowerShell arrays use commas, for example
`-Conditions 1,3 -Subjects 122,314`. Native CLI subject arguments are emitted as
space-separated integers, with one subject per command. Subject order follows
the condition configs; condition order follows `-Conditions`.

## Outputs And Failures

Each subject has a separate directory:

```text
OUTPUT/cond1/subject_122/fit/subject_122/
OUTPUT/cond1/subject_122/evaluate/subject_122/
OUTPUT/cond1/subject_122/png/
OUTPUT/logs/cond1_subject_122_fit.log
OUTPUT/logs/cond1_subject_122_evaluate.log
OUTPUT/logs/cond1_subject_122_plot.log
OUTPUT/plan.json
OUTPUT/progress.log
```

`plan.json` records programs, exact argument arrays, output locations, worker
count, and smoke status. Native outputs retain their existing scientific
configuration and provenance. `progress.log` is UTF-8 and records timestamped
START/DONE entries, then COMPLETE or FAILED. Per-stage logs retain streamed
command output; their text encoding follows the PowerShell version's
`Tee-Object` default. Console output shows the same progress and command output.

The first command failure stops the batch. Completed and partial outputs remain
available for inspection. Rerun remaining subjects into a new output root; the
script always repeats all three stages for each requested subject. For example,
after inspecting progress and determining that only 131 and 132 remain in Cond1:

```powershell
& './tools/run_cond123_local.ps1' -Conditions 1 -Subjects 131,132 -OutputRoot 'results/cond1_remaining_new'
```

## Runner Tests

From the repository root, using Python with PyYAML available:

```powershell
python -B -m unittest discover -s tests -p test_local_pipeline.py -v
```

The tests check CSV subject coverage, unchanged fit budgets, full-trial defaults,
ordered dry plans, selections, paths with spaces, fresh-output refusal, dependency
preflight (including a real Python import failure before output creation), command
logs, and immediate stopping after mocked failures at each stage. Mock executions
request two workers; no native fit is started by this
test file. The real three-subject (S122/S206/S314), four-trial script integration
check completed all nine stages using two workers, producing nine PNGs under
`results/local_cond123_smoke`. A separate four-trial tiny evaluation accepted
`--jobs 12`; this is not a full CPU utilization or scaling benchmark.

The complete Python suite passed 28 tests and 13 subtests. The native suite
passed 34 tests and 18,955 assertions after rebuilding with the 12-worker cap.
The default dry plan selects 96 subjects and 288 stage commands; the new full
96-subject batch has not been launched by this preparation task.

Keep the PowerShell session running and disable sleep for a long local batch.
All outputs are local; there is no background service or automatic remote sync.
