#!/usr/bin/env bash
# One-shot installer for the Lichtfeld-HYWorld2-Plugin on Linux / macOS.
#
# Creates a single symlink:
#   <LFS plugins>/hyworld2_plugin   -> this plugin directory
#
# The plugin is fully self-contained: the `hyworld2` Python package is
# vendored at ./hyworld2/, and the skyseg ONNX is cached on first use at
# ~/.cache/hyworld2_plugin/.
#
# Usage:
#   ./install.sh
#   LFS_PLUGINS_DIR=~/.lichtfeld/plugins ./install.sh
#   FORCE=1 ./install.sh                   # replace non-link targets

set -euo pipefail

plugin_root="$(cd "$(dirname "$0")" && pwd)"
lfs_plugins_dir="${LFS_PLUGINS_DIR:-$HOME/.lichtfeld/plugins}"

if [[ ! -f "$plugin_root/hyworld2/worldrecon/pipeline.py" ]]; then
    echo "Vendored hyworld2 package not found at $plugin_root/hyworld2/." >&2
    echo "This install is corrupt - re-clone the plugin repo." >&2
    exit 1
fi

plugin_link="$lfs_plugins_dir/hyworld2_plugin"

replace_symlink() {
    local link="$1" target="$2"
    mkdir -p "$(dirname "$link")"
    if [[ -L "$link" ]]; then
        rm -f "$link"
    elif [[ -e "$link" ]]; then
        if [[ "${FORCE:-}" == "1" ]]; then
            rm -rf "$link"
        else
            echo "Refusing to replace non-link: $link  (set FORCE=1 to override)" >&2
            exit 1
        fi
    fi
    ln -s "$target" "$link"
}

echo ""
echo "== Lichtfeld-HYWorld2-Plugin installer =="
echo "Plugin root:     $plugin_root"
echo "LFS plugins dir: $lfs_plugins_dir"

echo ""
echo "Creating symlink..."
replace_symlink "$plugin_link" "$plugin_root"
echo "  [OK] $plugin_link"
echo "       -> $plugin_root"

echo ""
echo "Install complete."

# Linux-only preflight: gsplat JIT-compiles CUDA kernels on first import and
# needs both `nvcc` (bundled via the `nvidia-cuda-nvcc` pip wheel — installed
# automatically by `uv sync`) and `gcc` as the host compiler. The host gcc
# is NOT in the venv and must be installed system-wide.
if [[ "$(uname)" == "Linux" ]]; then
    if ! command -v gcc >/dev/null 2>&1; then
        echo ""
        echo "WARNING: 'gcc' not found in PATH."
        echo "  gsplat will fail to build its CUDA kernels on first run."
        echo "  Fix with:"
        echo "      sudo apt install build-essential        # Debian/Ubuntu"
        echo "      sudo dnf install gcc-c++ make           # Fedora/RHEL"
        echo "      sudo pacman -S base-devel               # Arch"
    fi
fi

echo ""
echo "Next steps:"
echo "  1. Launch LichtFeld Studio."
echo "  2. Wait for first-run 'uv sync' (heavy CUDA wheels)."
echo "  3. Open the 'HY-World-Mirror-2' panel and run a reconstruction."
echo ""
echo "To uninstall: ./uninstall.sh"
