#!/usr/bin/env bash
# Phase-0 WSL checks for AAS transport bridge. Run inside WSL:
#   bash /mnt/f/UAM/scripts/wsl_phase0_check.sh
set +e

echo "=== uname ==="
uname -a

echo "=== nvidia-smi ==="
nvidia-smi | head -20 || echo "nvidia-smi FAILED"

# idempotent env for Gazebo / WSLg
BRC="${HOME}/.bashrc"
add_line() {
  local line="$1"
  grep -qxF "$line" "$BRC" 2>/dev/null || echo "$line" >> "$BRC"
}
add_line 'export GALLIUM_DRIVER=d3d12'
add_line 'export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA'
add_line 'export DISPLAY=${DISPLAY:-:0}'
add_line 'export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}'
add_line 'export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}'
# shellcheck disable=SC1090
source "$BRC" || true

echo "=== bashrc GPU lines ==="
grep -E 'GALLIUM|MESA_D3D12|XDG_RUNTIME' "$BRC" || true

echo "=== glxinfo ==="
if ! command -v glxinfo >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq mesa-utils || true
fi
glxinfo 2>/dev/null | grep -i 'OpenGL renderer' || echo "glxinfo FAILED (llvmpipe or missing)"

echo "=== docker ==="
DOCKER_OK=0
if command -v docker >/dev/null 2>&1; then
  docker --version
  if docker ps >/dev/null 2>&1; then
    echo "docker OK"
    DOCKER_OK=1
  else
    echo "docker present but daemon not reachable"
  fi
else
  echo "docker MISSING — enable Docker Desktop WSL integration for this distro"
fi

echo "=== AAS ==="
AAS_ROOT="${AAS_ROOT:-$HOME/aerial-autonomy-stack}"
AAS_OK=0
if [[ -d "$AAS_ROOT/tools_and_docs" ]]; then
  echo "AAS OK at $AAS_ROOT"
  AAS_OK=1
else
  echo "AAS MISSING at $AAS_ROOT — clone JacopoPan/aerial-autonomy-stack per REQUIREMENTS_WSL.md"
fi

echo "=== UAM ==="
UAM_ROOT="${UAM_ROOT:-/mnt/f/UAM}"
ls -d "$UAM_ROOT" && echo "UAM OK"

echo "=== python ==="
python3 -c 'import numpy; print("numpy", numpy.__version__)'
python3 -c 'import torch; print("torch", torch.__version__)' 2>/dev/null || echo "torch MISSING (dry_run uses numpy fallback)"

echo "=== summary ==="
echo "nvidia=OK gallium_env=OK docker=$DOCKER_OK aas=$AAS_OK"
echo "Live SITL needs docker=1 aas=1; otherwise use DRY_RUN=1"
echo "phase0_check: done"
exit 0
