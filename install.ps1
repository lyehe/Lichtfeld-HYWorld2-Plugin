#requires -Version 5.1
<#
.SYNOPSIS
  One-shot installer for the Lichtfeld-HYWorld2-Plugin.

.DESCRIPTION
  Creates a single directory junction (no admin needed):

    <LFS plugins>/hyworld2_plugin  -> this plugin directory

  The plugin is fully self-contained: the `hyworld2` Python package is
  vendored at ./hyworld2/ inside this directory, and the skyseg ONNX model
  is downloaded on first use into ~/.cache/hyworld2_plugin/.

  After this runs, launch LichtFeld Studio — it will `uv sync` the plugin
  venv on first load (torch 2.11 cu130, gsplat, onnxruntime-gpu, …) and
  pull WorldMirror weights lazily on the first reconstruction.

.PARAMETER LFSPluginsDir
  LichtFeld Studio plugins dir. Default: $env:USERPROFILE\.lichtfeld\plugins.

.PARAMETER Force
  Replace an existing target at <LFS plugins>/hyworld2_plugin even if it's
  not a link.
#>

[CmdletBinding()]
param(
    [string]$LFSPluginsDir,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$PluginRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $LFSPluginsDir) {
    $LFSPluginsDir = Join-Path $env:USERPROFILE ".lichtfeld\plugins"
}

function Remove-LinkLike([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    $item = Get-Item -LiteralPath $path -Force
    $isLink = ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq [IO.FileAttributes]::ReparsePoint
    if ($isLink) {
        & cmd /c rmdir (Get-Item -LiteralPath $path).FullName | Out-Null
        return
    }
    if (-not $Force) {
        throw "Path exists and is not a link: $path  (use -Force to remove)"
    }
    Remove-Item -LiteralPath $path -Recurse -Force
}

function New-DirectoryJunction([string]$link, [string]$target) {
    $linkParent = Split-Path -Parent $link
    if (-not (Test-Path $linkParent)) {
        New-Item -ItemType Directory -Path $linkParent -Force | Out-Null
    }
    Remove-LinkLike $link
    try {
        New-Item -ItemType Junction -Path $link -Target $target -ErrorAction Stop | Out-Null
    } catch {
        $result = & cmd /c mklink /J "$link" "$target" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create junction '$link' -> '$target': $result"
        }
    }
}

# ------------------------------------------------------------------- Go
Write-Host ""
Write-Host "== Lichtfeld-HYWorld2-Plugin installer ==" -ForegroundColor Cyan
Write-Host "Plugin root:     $PluginRoot"
Write-Host "LFS plugins dir: $LFSPluginsDir"

$vendored = Join-Path $PluginRoot "hyworld2\worldrecon\pipeline.py"
if (-not (Test-Path $vendored -PathType Leaf)) {
    throw "Vendored hyworld2 package not found at $vendored. This install is corrupt — re-clone the plugin repo."
}

$pluginLink = Join-Path $LFSPluginsDir "hyworld2_plugin"

Write-Host ""
Write-Host "Creating junction..."
New-DirectoryJunction -link $pluginLink -target $PluginRoot
Write-Host "  [OK] $pluginLink" -ForegroundColor Green
Write-Host "       -> $PluginRoot" -ForegroundColor DarkGray

Write-Host ""
Write-Host "Install complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Launch LichtFeld Studio."
Write-Host "  2. Wait for first-run 'uv sync' (downloads torch cu130 / gsplat / onnxruntime-gpu)."
Write-Host "  3. Open the 'HY-World-Mirror-2' panel and run a reconstruction."
Write-Host ""
Write-Host "To uninstall: .\uninstall.ps1" -ForegroundColor DarkGray
