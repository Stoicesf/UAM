#!/usr/bin/env bash
# 4-quad + transport bridge live SITL (headless).
set -euo pipefail
export UAM_ROOT="${UAM_ROOT:-/mnt/f/UAM}"
export AAS_ROOT="${AAS_ROOT:-$HOME/aerial-autonomy-stack}"
export UAV_COUNT=4
export HEADLESS=true
export STEPS=80
export ALTITUDE=12
export TARGET_X=5.0
export TARGET_Y=0.0
export KEEPALIVE_SEC=1500
export GALLIUM_DRIVER=d3d12
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
export PYTHONPATH="$UAM_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Ensure docker daemon
if ! docker info >/dev/null 2>&1; then
  sudo service docker start || true
  sleep 2
fi

bash "$UAM_ROOT/scripts/run_aas_transport.sh" 2>&1 | tee /tmp/aas_transport_live.log
echo "live transport rc=${PIPESTATUS[0]}"
