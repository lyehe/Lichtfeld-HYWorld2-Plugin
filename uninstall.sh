#!/usr/bin/env bash
# Remove the symlink created by install.sh.
set -euo pipefail

lfs_plugins_dir="${LFS_PLUGINS_DIR:-$HOME/.lichtfeld/plugins}"
plugin_link="$lfs_plugins_dir/hyworld2_plugin"

plugin_root="$(cd "$(dirname "$0")" && pwd)"
models_dir="$plugin_root/models"

echo ""
echo "== Lichtfeld-HYWorld2-Plugin uninstaller =="
if [[ -L "$plugin_link" ]]; then
    rm -f "$plugin_link"
    echo "  [rm]   $plugin_link"
elif [[ -e "$plugin_link" ]]; then
    echo "  [skip] $plugin_link (not a symlink; leaving alone)"
else
    echo "  [skip] $plugin_link (not present)"
fi
for sub in models cache; do
    d="$plugin_root/$sub"
    if [[ -d "$d" ]]; then
        echo "  [rm]   $d"
        rm -rf "$d"
    else
        echo "  [skip] $d (not present)"
    fi
done
echo ""
echo "Done. Plugin venv (.venv/) is not removed - delete manually if desired."
