$root = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $root ".cache\uv"
$env:PIP_CACHE_DIR = Join-Path $root ".cache\pip"
$env:HF_HOME = Join-Path $root ".cache\huggingface"
$env:HF_HUB_CACHE = Join-Path $root ".cache\huggingface\hub"
$env:TORCH_HOME = Join-Path $root ".cache\torch"
$env:XDG_CACHE_HOME = Join-Path $root ".cache\xdg"
$env:MPLCONFIGDIR = Join-Path $root ".cache\matplotlib"
Write-Output "Local caches enabled for this PowerShell session."
