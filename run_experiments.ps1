# ============================================================
# run_experiments.ps1 — Early Exit Architecture Research Runner
# ============================================================
# Usage:
#   .\run_experiments.ps1                   # Default: architecture ablation only
#   .\run_experiments.ps1 -All             # Run all experiments
#   .\run_experiments.ps1 -Ablation        # Architecture ablation
#   .\run_experiments.ps1 -Sizes           # Model size scaling
#   .\run_experiments.ps1 -Pruning         # Structured pruning
#   .\run_experiments.ps1 -Strategies      # Threshold strategy comparison
#   .\run_experiments.ps1 -Tuning          # Hyperparameter search
#   .\run_experiments.ps1 -Trials 5        # Custom trial count
#   .\run_experiments.ps1 -Seed 0          # Custom seed
#   .\run_experiments.ps1 -Strategy entropy # Threshold strategy
# ============================================================

param(
    [switch]$All,
    [switch]$Ablation,
    [switch]$Sizes,
    [switch]$Pruning,
    [switch]$Strategies,
    [switch]$Tuning,
    [int]$Trials = 3,
    [int]$Seed = 42,
    [ValidateSet("confidence","entropy","patience")]
    [string]$Strategy = "confidence"
)

$ProjectDir = $PSScriptRoot
Set-Location $ProjectDir

# ── Pretty banner ────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Early Exit Architecture Research — Experiment Runner" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Project: $ProjectDir" -ForegroundColor Gray
Write-Host "  Seed   : $Seed   Trials: $Trials   Strategy: $Strategy" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Verify Python is available ───────────────────────────────
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python not found. Please install Python 3.8+ and add it to PATH." -ForegroundColor Red
    exit 1
}

# ── Verify key files exist ───────────────────────────────────
$RequiredFiles = @("main.py", "dataset.py", "models.py", "train.py", "evaluate.py", "analysis.py", "visualize.py")
foreach ($f in $RequiredFiles) {
    if (-not (Test-Path "$ProjectDir\$f")) {
        Write-Host "[ERROR] Missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

# ── Build argument string ────────────────────────────────────
$Args = "--seed $Seed --trials $Trials --strategy $Strategy"

if ($All)        { $Args += " --run-all" }
if ($Ablation)   { $Args += " --run-ablation" }
if ($Sizes)      { $Args += " --run-sizes" }
if ($Pruning)    { $Args += " --run-pruning" }
if ($Strategies) { $Args += " --run-strategies" }
if ($Tuning)     { $Args += " --run-tuning" }

# If nothing selected, default to ablation
if (-not ($All -or $Ablation -or $Sizes -or $Pruning -or $Strategies -or $Tuning)) {
    Write-Host "[INFO] No experiment flag given — defaulting to --run-ablation" -ForegroundColor Yellow
    $Args += " --run-ablation"
}

# ── Run ──────────────────────────────────────────────────────
$StartTime = Get-Date
Write-Host "[RUN] python main.py $Args" -ForegroundColor Green
Write-Host ""

$Process = Start-Process -FilePath "python" `
    -ArgumentList "main.py $Args" `
    -WorkingDirectory $ProjectDir `
    -NoNewWindow -PassThru -Wait

$Duration = (Get-Date) - $StartTime
$Minutes  = [int]$Duration.TotalMinutes
$Seconds  = $Duration.Seconds

Write-Host ""
if ($Process.ExitCode -eq 0) {
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  ✓ All experiments finished in ${Minutes}m ${Seconds}s" -ForegroundColor Green
    Write-Host "  Plots  → $ProjectDir\plots\" -ForegroundColor Green
    Write-Host "  Results→ $ProjectDir\results\" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "  ✗ Pipeline exited with code $($Process.ExitCode)" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    exit $Process.ExitCode
}

# ── Optional: open plots folder ──────────────────────────────
$Open = Read-Host "`nOpen plots folder? (y/N)"
if ($Open -match "^[Yy]") {
    Invoke-Item "$ProjectDir\plots"
}
