#!/usr/bin/env bash
# Headless AAS + theory transport bridge (no Gazebo GUI).
#   bash /mnt/f/UAM/scripts/run_theory_transport_headless.sh
# Dry-run only:
#   DRY_RUN=1 bash /mnt/f/UAM/scripts/run_theory_transport_headless.sh

set -euo pipefail
export UAM_ROOT="${UAM_ROOT:-/mnt/f/UAM}"
export AAS_ROOT="${AAS_ROOT:-$HOME/aerial-autonomy-stack}"
export UAV_COUNT="${UAV_COUNT:-4}"
export HEADLESS=true
export STEPS="${STEPS:-80}"
export ALTITUDE="${ALTITUDE:-12}"
export TARGET_X="${TARGET_X:-5.0}"
export TARGET_Y="${TARGET_Y:-0.0}"
export KEEPALIVE_SEC="${KEEPALIVE_SEC:-1500}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
export PYTHONPATH="$UAM_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYBRID="${HYBRID:-1}"
export RISE="${RISE:-1}"

if [[ "${DRY_RUN:-0}" != "1" ]] && ! docker info >/dev/null 2>&1; then
  sudo service docker start || true
  sleep 2
fi

bash "$UAM_ROOT/scripts/run_theory_transport.sh" "$TARGET_X" "$TARGET_Y" 2>&1 | tee /tmp/theory_transport_headless.log
echo "theory transport headless rc=${PIPESTATUS[0]}"
