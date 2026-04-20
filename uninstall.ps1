#requires -Version 5.1
<#
.SYNOPSIS
  Remove the junction created by install.ps1.
#>
[CmdletBinding()]
param(
    [string]$LFSPluginsDir
)
$ErrorActionPreference = "Stop"

if (-not $LFSPluginsDir) {
    $LFSPluginsDir = Join-Path $env:USERPROFILE ".lichtfeld\plugins"
}

function Remove-LinkIfPresent([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host "  [skip] $path (not present)" -ForegroundColor DarkGray
        return
    }
    $item = Get-Item -LiteralPath $path -Force
    $isLink = ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq [IO.FileAttributes]::ReparsePoint
    if (-not $isLink) {
        Write-Host "  [skip] $path (not a junction/symlink; leaving alone)" -ForegroundColor Yellow
        return
    }
    & cmd /c rmdir (Get-Item -LiteralPath $path).FullName | Out-Null
    Write-Host "  [rm]   $path" -ForegroundColor Green
}

Write-Host ""
Write-Host "== Lichtfeld-HYWorld2-Plugin uninstaller ==" -ForegroundColor Cyan
Remove-LinkIfPresent (Join-Path $LFSPluginsDir "hyworld2_plugin")

# Remove the plugin-local model cache (WorldMirror weights + skyseg ONNX).
$PluginRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
foreach ($sub in @("models", "cache")) {
    $dir = Join-Path $PluginRoot $sub
    if (Test-Path $dir) {
        Write-Host "  [rm]   $dir" -ForegroundColor Green
        Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "  [skip] $dir (not present)" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "Done. Plugin venv (.venv/) is not removed - delete manually if desired." -ForegroundColor DarkGray
