# Task 2: SCE probe training + scalar baselines + analysis
# Run from the project root (vessel/)

Set-Location $PSScriptRoot\..

python -m src.main.train_sce_probe --config configs/sce_probe.yaml
