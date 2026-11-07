[CmdletBinding()]
param(
    [ValidateRange(1, 12)][int]$Jobs = 12,
    [string]$PythonExe = 'python',
    [string]$Executable = 'build/Release/genmodel_cpp.exe',
    [string]$ConfigDirectory = 'configs',
    [string]$OutputRoot,
    [ValidateSet(1, 2, 3)][int[]]$Conditions = @(1, 2, 3),
    [ValidateRange(1, 2147483647)][int]$MaxTrials,
    [int[]]$Subjects,
    [switch]$Smoke,
    [switch]$ExportPaper,
    [switch]$PlanOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

function Resolve-RepoPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $repo $Path))
}

function Resolve-Program([string]$Name) {
    if ([IO.Path]::IsPathRooted($Name) -or $Name.Contains('/') -or $Name.Contains('\')) {
        return Resolve-RepoPath $Name
    }
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    if ($PlanOnly) { return $Name }
    throw "Program not found: $Name"
}

if (-not $Conditions.Count -or @($Conditions | Select-Object -Unique).Count -ne $Conditions.Count) {
    throw 'Conditions must be nonempty and unique.'
}
if ($PSBoundParameters.ContainsKey('Subjects') -and
    (-not $Subjects.Count -or @($Subjects | Select-Object -Unique).Count -ne $Subjects.Count)) {
    throw 'Subjects must be nonempty and unique.'
}

$native = Resolve-Program $Executable
$python = Resolve-Program $PythonExe
$plotter = Join-Path $repo 'evaluation/plot_results.py'
$paperExporter = Join-Path $repo 'evaluation/export_paper_tables.py'
$configRoot = Resolve-RepoPath $ConfigDirectory
if (-not $OutputRoot) {
    $OutputRoot = 'results/local_cond123_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
}
$output = Resolve-RepoPath $OutputRoot
if (Test-Path -LiteralPath $output) { throw "OutputRoot already exists: $output" }

# Read the explicit config lists without requiring Python during dry planning.
$selected = @()
foreach ($condition in $Conditions) {
    $config = Join-Path $configRoot "cond${condition}_full_local.yaml"
    $subjectLine = @(Get-Content -LiteralPath $config | Where-Object { $_ -match '^subjects:' })
    if ($subjectLine.Count -ne 1 -or $subjectLine[0] -notmatch '^subjects:\s*\[([0-9, ]+)\]\s*$') {
        throw "Expected an explicit integer subject list in $config"
    }
    $ids = @($Matches[1].Split(',') | ForEach-Object { [int]$_.Trim() })
    foreach ($id in $ids) {
        if (-not $PSBoundParameters.ContainsKey('Subjects') -or $Subjects -contains $id) {
            $selected += [pscustomobject]@{ condition = $condition; subject = $id; config = $config }
        }
    }
}
if ($PSBoundParameters.ContainsKey('Subjects')) {
    foreach ($id in $Subjects) {
        if ($selected.subject -notcontains $id) { throw "Subject $id is absent from the selected conditions." }
    }
}
if (-not $selected.Count) { throw 'No subjects selected.' }

$steps = @()
foreach ($item in $selected) {
    $base = Join-Path $output "cond$($item.condition)/subject_$($item.subject)"
    $fit = Join-Path $base 'fit'
    $evaluation = Join-Path $base 'evaluate'
    $common = @('--config', $item.config, '--subjects', [string]$item.subject, '--jobs', [string]$Jobs)
    if ($PSBoundParameters.ContainsKey('MaxTrials')) { $common += @('--max-trials', [string]$MaxTrials) }
    if ($Smoke) { $common += '--smoke' }
    $stages = @('fit', 'evaluate', 'plot')
    if ($ExportPaper) { $stages += 'export-paper' }
    foreach ($stage in $stages) {
        if ($stage -eq 'fit') { $program = $native; $arguments = @('fit') + $common + @('--output', $fit) }
        elseif ($stage -eq 'evaluate') {
            $program = $native
            $arguments = @('evaluate') + $common + @('--fit-dir', $fit, '--output', $evaluation)
        } elseif ($stage -eq 'plot') {
            $program = $python
            $arguments = @('-B', $plotter, '--input', (Join-Path $evaluation "subject_$($item.subject)"),
                           '--output', (Join-Path $base 'png'))
        } else {
            $program = $python
            $arguments = @('-B', $paperExporter, '--input', (Join-Path $evaluation "subject_$($item.subject)"),
                           '--output', (Join-Path $base 'paper'), '--window', '16')
        }
        $steps += [pscustomobject]@{ condition = $item.condition; subject = $item.subject;
            stage = $stage; program = $program; arguments = $arguments }
    }
}
$plan = [ordered]@{ output_root = $output; jobs = $Jobs; smoke = [bool]$Smoke;
    config_directory = $configRoot; export_paper = [bool]$ExportPaper; steps = $steps }
if ($PlanOnly) { $plan | ConvertTo-Json -Depth 8; return }
foreach ($program in @($native, $python, $plotter)) {
    if (-not (Test-Path -LiteralPath $program -PathType Leaf)) { throw "Missing program/file: $program" }
}
if ($ExportPaper -and -not (Test-Path -LiteralPath $paperExporter -PathType Leaf)) {
    throw "Missing paper exporter: $paperExporter"
}

$progress = Join-Path $output 'progress.log'
function Write-ProgressLine([string]$Message) {
    $line = (Get-Date -Format o) + ' ' + $Message
    Add-Content -LiteralPath $progress -Value $line -Encoding UTF8
    Write-Host $line
}

$threadVariables = @('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_MAX_ACTIVE_LEVELS')
$saved = @{}
foreach ($name in $threadVariables) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, '1', 'Process')
}
$index = 0
$runStarted = $false
try {
    Push-Location -LiteralPath $repo
    try {
        # Probe with the same working directory and thread bounds as offline plotting.
        $global:LASTEXITCODE = 0
        $ErrorActionPreference = 'Continue'
        try { & $python -B -c 'import numpy; import scipy; import matplotlib' 2>&1 | Out-Host }
        finally { $ErrorActionPreference = 'Stop' }
        if ($LASTEXITCODE -ne 0) {
            throw "Python dependency preflight failed (exit code $LASTEXITCODE): $python requires numpy, scipy, matplotlib. No run output was created."
        }

        # Reserve only a fresh root. Failed partial results remain available for inspection.
        $null = New-Item -ItemType Directory -Path $output
        $null = New-Item -ItemType Directory -Path (Join-Path $output 'logs')
        $plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'plan.json') -Encoding UTF8
        $runStarted = $true
        foreach ($step in $steps) {
            $index++
            $label = "[$index/$($steps.Count)] cond$($step.condition) subject_$($step.subject) $($step.stage)"
            $log = Join-Path $output "logs/cond$($step.condition)_subject_$($step.subject)_$($step.stage).log"
            Write-ProgressLine "START $label"
            $global:LASTEXITCODE = 0
            $arguments = $step.arguments
            # PS 5.1 exposes native stderr as ErrorRecords; retain it without terminating early.
            $ErrorActionPreference = 'Continue'
            try { & $step.program @arguments 2>&1 | Tee-Object -FilePath $log | Out-Host }
            finally { $ErrorActionPreference = 'Stop' }
            if ($LASTEXITCODE -ne 0) { throw "Exit code $LASTEXITCODE at $label; see $log" }
            Write-ProgressLine "DONE $label"
        }
        Write-ProgressLine 'COMPLETE'
    } finally { Pop-Location }
} catch {
    if ($runStarted) { Write-ProgressLine "FAILED $($_.Exception.Message)" }
    throw
} finally {
    foreach ($name in $threadVariables) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
