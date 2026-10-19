# Install the paired code/data release inside this checkout.
[CmdletBinding()]
param([string]$Adata = "", [string]$DataDir = "", [switch]$ExecuteFigures, [switch]$Help)
$ErrorActionPreference = "Stop"
if ($Help) {
    Write-Host 'Usage: .\setup_dev.ps1 [-DataDir DIR] [-Adata FILE] [-ExecuteFigures]'
    exit 0
}
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv } else { python -m venv .venv }
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
& $Python -m pip install -c constraints.txt -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
$DataArgs = @()
if ($Adata) { $DataArgs += @('--study-source', $Adata) }
if ($DataDir) { $DataArgs += @('--source-dir', $DataDir) }
& $Python tools/fetch_release_data.py @DataArgs
if ($LASTEXITCODE -ne 0) { throw 'Release data installation or verification failed' }
if ($ExecuteFigures) {
    $env:PYTHONPATH = $RepoRoot
    & $Python tools/run_all_notebooks.py
    if ($LASTEXITCODE -ne 0) { throw 'Notebook execution failed' }
}
Write-Host 'Setup complete. Activate with: .\.venv\Scripts\Activate.ps1'
Write-Host 'Open notebooks with: cd notebooks; jupyter notebook'
