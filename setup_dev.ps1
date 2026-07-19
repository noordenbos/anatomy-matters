# Bootstrap a local checkout for figure reproduction (Windows PowerShell).
#
# Recommended (Zenodo download when published):
#   .\setup_dev.ps1
#
# Workaround until Zenodo is live - point at a local AnnData file (symlink, no copy):
#   .\setup_dev.ps1 -Adata C:\path\to\DLBCL_location_2026.h5ad
#
# Optional: run all figure notebooks after setup:
#   .\setup_dev.ps1 -ExecuteFigures -Adata C:\path\to\DLBCL_location_2026.h5ad
#
# If script execution is blocked once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# Override Zenodo record without editing this file:
#   $env:ZENODO_RECORD_ID = "12345678"; .\setup_dev.ps1

[CmdletBinding()]
param(
    [string]$Adata = "",
    [switch]$ExecuteFigures,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Set this when the AnnData bundle is published on Zenodo (digits only).
# Leave empty to require -Adata until then.
$DefaultZenodoRecordId = ""

$AdataBasename = "DLBCL_location_2026.h5ad"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Show-Usage {
    @"
Usage: .\setup_dev.ps1 [-Adata PATH] [-ExecuteFigures]

Options:
  -Adata PATH        Symlink PATH into data\DLBCL_location_2026.h5ad
  -ExecuteFigures    Run all figure notebooks after setup
  -Help              Show this help

If no local AnnData is given, the script downloads from Zenodo when
ZENODO_RECORD_ID is set (env) or DefaultZenodoRecordId in this file.
"@
}

if ($Help) {
    Show-Usage
    exit 0
}

function Get-PythonLauncher {
    foreach ($cmd in @("py", "python", "python3")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($null -ne $found) {
            if ($cmd -eq "py") {
                return @{ Exe = "py"; Prefix = @("-3") }
            }
            return @{ Exe = $cmd; Prefix = @() }
        }
    }
    throw "Python not found. Install Python 3.10+ from https://www.python.org/downloads/ (enable 'Add python.exe to PATH')."
}

function Test-AdataReady {
    param([string]$LinkPath)
    return (Test-Path -LiteralPath $LinkPath)
}

function Link-Adata {
    param([string]$Src, [string]$LinkPath)
    if (-not (Test-Path -LiteralPath $Src -PathType Leaf)) {
        throw "AnnData file not found: $Src"
    }
    $Src = (Resolve-Path -LiteralPath $Src).Path
    if (Test-Path -LiteralPath $LinkPath) {
        Write-Host "AnnData already present at $LinkPath (leaving as-is)"
        return
    }
    $linkDir = Split-Path -Parent $LinkPath
    if (-not (Test-Path -LiteralPath $linkDir)) {
        New-Item -ItemType Directory -Force -Path $linkDir | Out-Null
    }
    try {
        New-Item -ItemType SymbolicLink -Path $LinkPath -Target $Src | Out-Null
        Write-Host "Symlinked $LinkPath -> $Src"
    } catch {
        Write-Host "Symbolic link failed ($($_.Exception.Message)); copying instead (slower / uses disk) ..."
        Copy-Item -LiteralPath $Src -Destination $LinkPath
        Write-Host "Copied $Src -> $LinkPath"
    }
}

function Download-AdataFromZenodo {
    param([string]$RecordId, [string]$LinkPath)
    $url = "https://zenodo.org/records/$RecordId/files/$AdataBasename`?download=1"
    Write-Host "Downloading AnnData from Zenodo (record $RecordId) ..."
    Write-Host "  $url"
    $linkDir = Split-Path -Parent $LinkPath
    if (-not (Test-Path -LiteralPath $linkDir)) {
        New-Item -ItemType Directory -Force -Path $linkDir | Out-Null
    }
    Invoke-WebRequest -Uri $url -OutFile $LinkPath
    Write-Host "Saved $LinkPath"
}

function Invoke-AllFigureNotebooks {
    param([string]$LinkPath)
    if (-not (Test-AdataReady $LinkPath)) {
        throw "Cannot run notebooks without $LinkPath"
    }
    Write-Host "Executing figure notebooks (most finish in under ~5 minutes each; full suite often ~10-20 minutes) ..."
    $env:PYTHONPATH = $RepoRoot
    & python tools/run_all_notebooks.py
    if ($LASTEXITCODE -ne 0) { throw "run_all_notebooks.py failed (exit $LASTEXITCODE)" }
}

function Start-JupyterNotebooks {
    param([string]$LinkPath)
    if (-not (Test-AdataReady $LinkPath)) {
        throw "Cannot open notebooks without $LinkPath"
    }
    Write-Host "Starting Jupyter in notebooks/ (Ctrl+C to stop the server) ..."
    Push-Location notebooks
    try {
        & jupyter notebook
    } finally {
        Pop-Location
    }
}

function Write-Completion {
    param([string]$LinkPath)
    Write-Host ""
    Write-Host "Setup complete."
    Write-Host ""
    Write-Host "Next time, first activate the environment:"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host ""
    Write-Host "Then either:"
    Write-Host "  (1) Explore notebooks in the browser:"
    Write-Host "        cd notebooks; jupyter notebook"
    Write-Host "  (2) Re-run all notebooks (figures + tables):"
    Write-Host "        `$env:PYTHONPATH = `"$RepoRoot`"; python tools/run_all_notebooks.py"
    Write-Host ""
    if (Test-AdataReady $LinkPath) {
        Write-Host "AnnData: $LinkPath"
    } else {
        Write-Host "AnnData: missing - re-run with -Adata PATH (see above)"
    }
}

$py = Get-PythonLauncher
Write-Host "Using Python launcher: $($py.Exe) $($py.Prefix -join ' ')"

if (-not (Test-Path -LiteralPath ".venv")) {
    Write-Host "Creating virtual environment (.venv) ..."
    & $py.Exe @($py.Prefix + @("-m", "venv", ".venv"))
}

$activate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path -LiteralPath $activate)) {
    throw "Virtual environment activate script missing: $activate"
}
. $activate

python -m pip install --upgrade pip
pip install -r requirements.txt

$AdataLink = Join-Path "data" $AdataBasename
New-Item -ItemType Directory -Force -Path "data" | Out-Null

$zenodoRecordId = if ($env:ZENODO_RECORD_ID) { $env:ZENODO_RECORD_ID } else { $DefaultZenodoRecordId }

if ($Adata) {
    Link-Adata -Src $Adata -LinkPath $AdataLink
} elseif (Test-Path -LiteralPath $AdataLink) {
    Write-Host "AnnData already present at $AdataLink"
} elseif ($zenodoRecordId) {
    Download-AdataFromZenodo -RecordId $zenodoRecordId -LinkPath $AdataLink
} else {
    Write-Host @"

AnnData is not installed yet, and Zenodo download is not enabled
(DefaultZenodoRecordId is empty / ZENODO_RECORD_ID unset).

Workaround - link a local copy (no file copy when symlink works; ~0.6-2 GB):

  .\setup_dev.ps1 -Adata C:\path\to\$AdataBasename

When the Zenodo record is published, either set DefaultZenodoRecordId
in setup_dev.ps1 or run:

  `$env:ZENODO_RECORD_ID = "<record_id>"; .\setup_dev.ps1

"@
}

if ($ExecuteFigures) {
    Invoke-AllFigureNotebooks -LinkPath $AdataLink
} elseif ([Environment]::UserInteractive -and (Test-AdataReady $AdataLink) -and -not [Console]::IsInputRedirected) {
    Write-Host ""
    Write-Host "What next?"
    Write-Host "  1) Explore notebooks interactively in the Jupyter browser"
    Write-Host "  2) Run all notebooks and recreate figures and data tables locally"
    Write-Host "  3) Exit"
    Write-Host ""
    $choice = Read-Host "Enter choice [1/2/3]"
    switch -Regex ($choice.Trim()) {
        "^1$" { try { Start-JupyterNotebooks -LinkPath $AdataLink } catch { Write-Host $_ } }
        "^2$" { try { Invoke-AllFigureNotebooks -LinkPath $AdataLink } catch { Write-Host $_ } }
        "^(3)?$" { }
        default { Write-Host "Unrecognized choice ('$choice'); continuing." }
    }
}

Write-Completion -LinkPath $AdataLink
