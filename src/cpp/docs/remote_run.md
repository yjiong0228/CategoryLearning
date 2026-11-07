> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Bounded Remote Run

## Deployment And Current Status

Source, configs, rule resources, tests and documentation were uploaded to
`~/ran/CategoryLearning1031cpp_records_20261104` on 2026-11-04.
The source-only archive SHA256 is
`b927b508d42e146c5fb4e511f4582d9104f2cdc7efca2921f86ab0206ac014e5`.
Existing results and the entire participant `data/` tree were excluded. Three
input CSVs require separate transfer authorization; existing server Task2 bytes
differ even after accounting for line endings. Do not substitute the older inputs.
Linux compilation and scientific execution have not been performed. This deployment
keeps the original scientific repositories, tasks and results unchanged.
The runner performs local orchestration on the Linux server after deployment;
it performs no SSH, upload, installation, build or background launch.

The 2026-11-04 three-second server sample measured 127.933 busy CPU equivalents
out of an affinity allocation of 128, leaving approximately 0.0665 free.
No logical CPU was below 25% utilization. The full run is currently CPU-blocked.
There was about 28 TB free on `/data2`; the runner checks the output filesystem
again when execution is requested. This is a dated observation, not a reservation.

Completion within four hours remains unverified. Once resources are available,
a short benchmark is required to estimate runtime before the full run. The
deadline bounds execution; it does not establish that 96 subjects will finish.

## Plan And Foreground Execution

Run from the physical deployment path above. Linux execution requires the
compiled C++20/OpenMP executable at `build/genmodel_cpp`, the repository's data
and catalogue resources, and a Python environment with the existing offline
evaluation dependencies. The orchestration script itself uses only the Python
standard library. Model fitting, filtering and simulation run in the native
executable; Python runs the existing offline plot and table exporters.

Inspect the plan first. The output argument must name a fresh directory even
in plan mode; no output is created and resource sampling is skipped:

```bash
python -B tools/run_cond123_remote.py --plan-only --output results/remote_cond123_NEW
```

Only after resources and execution authorization are available, the foreground
command is:

```bash
python -B tools/run_cond123_remote.py --output results/remote_cond123_NEW --max-hours 4
```

`--subjects 122 206 314` selects a subset for a separately authorized check.
It preserves the full trial schedule and every configured budget. There is no
job-count override, force bypass, automatic retry, resume or budget reduction.
`--max-hours` accepts a finite positive value up to four. SIGINT, SIGTERM,
subprocess failure and deadline expiry stop the run. Timeout handling terminates
and kills the foreground subprocess and reaps it before returning. OS scheduling
and process reaping can add shutdown latency to the configured deadline.

## Resource Gate And Budgets

Before creating output or starting jobs, and between subjects, the runner reads
per-CPU `/proc/stat` counters twice, three seconds apart, within its CPU affinity.
Free CPU equivalents are the sum of each allowed CPU's idle-time fraction over
that interval. Guest time is already included in user/nice counters and is
counted once; iowait and steal time count as unavailable capacity. Missing or
invalid counters and affinity changes stop the run.

The gate requires at least 64 measured free CPU equivalents, an effective visible
cgroup CPU quota of at least 64 CPUs, and 100 GiB free on the output filesystem.
The quota check reads visible ancestor limits on cgroup v1 or v2 and fails when
it cannot determine the limits. These checks are availability samples, not an
exclusive CPU reservation. Other users can start work after a sample. Insufficient
resources stop the run; the runner does not wait in a queue or reduce budgets.
The 100 GiB threshold covers the expected uncompressed particle-packet storage
with headroom, subject to actual packet sizes and concurrent filesystem use.

The three `configs/cond{1,2,3}_full_remote.json` files are JSON-valid YAML,
with 32 subjects per condition, all trials, the original seed and parameter
supports, and paths relative to each config: `../data`, `../resources`, and
`parameter_space.yaml`. Budgets are:

| Stage | Budget |
| --- | --- |
| Native workers | 64; one subject at a time |
| Search | 64 starts, 3 rounds, 32 proposals per round, top 4; 48 particles, 4 seeds |
| Rescore | Top 4 plus 2 diverse candidates; 192 particles, 4 seeds |
| Audit | Top 2; 512 particles, 8 seeds |
| Evaluation | 512 particles, 32 PF repeats, 512 autonomous simulations, 512 band draws |
| Output | Window 16; plots and conditional particle export enabled |

The native implementation owns candidate-by-seed and evaluation parallelism.
The runner sets numeric-library thread counts to one and disables nested OpenMP
parallelism. It launches each command directly, without a shell or backgrounding.
The monotonic total deadline includes planning, gates, native work, validation,
plots and exports. Each subprocess receives only the remaining total time.

## Outputs And Validation

Each subject completes fit, evaluate, plot, then paper export before the next
subject starts. Within `condC/subject_ID/`, the outputs are:

| Path | Contents |
| --- | --- |
| `fit/subject_ID/` | Native fitted parameters and provenance |
| `evaluate/subject_ID/` | Actual native evaluation bundle and particle packet |
| `png/` | Existing offline plot outputs |
| `paper/` | Existing paper tables and PNGs, with window 16 |
| `fit.log`, `evaluate.log`, `plot.log`, `export-paper.log` | Combined per-stage stdout/stderr |

Both exporters read `evaluate/subject_ID/`. Their destinations are outside the
evaluation tree. The runner checks each new evaluation manifest's required
fields, subject/condition identity, configured budgets, seed counts and particle
conditioning metadata. It streams SHA256 checks of both NPZ files with bounded
memory before starting either exporter. These structural and integrity checks
do not establish numerical or scientific validity; the existing exporters add
their own array checks.

The run root retains `plan.json` with config paths, hashes, budgets and commands,
plus `progress.jsonl` with UTC times, remaining deadline, gate measurements,
stage completion and failures. Partial outputs remain available after failure;
every new attempt requires a new output directory.

The targeted test suite uses tiny synthetic packets and mocked subprocesses:

```bash
python -B -m pytest -q -p no:cacheprovider tests/test_remote_runner.py
```

No native model, full fit or remote execution is required by this suite.
