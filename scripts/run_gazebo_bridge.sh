#!/usr/bin/env bash
# Multi-UAV Gazebo/PX4 + AC-DSGF MAVROS bridge.
# Usage:
#   bash scripts/run_gazebo_bridge.sh [UAV_COUNT]
#   UAV_COUNT=4 bash scripts/run_gazebo_bridge.sh
#
# Env:
#   AAS_ROOT=~/aerial-autonomy-stack
#   UAM_ROOT=...   MODEL_PATH=...   MAX_VEL=2.0
#
# ponytail: prefers one Gazebo world + multi PX4 instances (AAS). Falling back
# to per-instance gnome-terminal loops if MULTI_LAUNCH unset.

set -euo pipefail

UAV_COUNT="${1:-${UAV_COUNT:-4}}"
AAS_ROOT="${AAS_ROOT:-$HOME/aerial-autonomy-stack}"
UAM_ROOT="${UAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-$UAM_ROOT/experiment_results/semantic/semantic_encoder.pt}"
MAX_VEL="${MAX_VEL:-2.0}"

if [[ ! -d "$AAS_ROOT" ]]; then
  echo "AAS_ROOT not found: $AAS_ROOT" >&2
  exit 1
fi

echo "[1/2] Starting $UAV_COUNT UAVs (AAS PX4 SITL + Gazebo)…"
cd "$AAS_ROOT"

# Prefer a single multi-vehicle launch if the Makefile target exists
if make -n px4_sitl_multi >/dev/null 2>&1; then
  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -c "cd '$AAS_ROOT' && make px4_sitl_multi NUM=$UAV_COUNT; exec bash"
  else
    make px4_sitl_multi "NUM=$UAV_COUNT" &
  fi
else
  # Fallback: staggered iris instances (-i). One shared gazebo if AAS supports it.
  for i in $(seq 1 "$UAV_COUNT"); do
    echo "  boot instance $i / $UAV_COUNT"
    if command -v gnome-terminal >/dev/null 2>&1; then
      gnome-terminal -- bash -c "cd '$AAS_ROOT' && make px4_sitl gazebo-classic HEADLESS:=1 PX4_SYS_AUTOSTART:=4001 -j1 2>/dev/null || make px4_sitl gazebo -i $i -m iris; exec bash" &
    else
      (cd "$AAS_ROOT" && make px4_sitl gazebo -i "$i" -m iris) &
    fi
    sleep 2
  done
fi

echo "Waiting 20s for Gazebo/MAVROS…"
sleep 20

echo "[2/2] Bridge N=$UAV_COUNT max_vel=$MAX_VEL"
export PYTHONPATH="${UAM_ROOT}:${PYTHONPATH:-}"
python3 "$UAM_ROOT/ros_nodes/ac_dsgf_bridge.py" --ros-args \
  -p "model_path:=$MODEL_PATH" \
  -p "uav_count:=$UAV_COUNT" \
  -p "namespace_prefix:=uav" \
  -p "shield_type:=cbf" \
  -p "max_vel:=$MAX_VEL"

echo "Gazebo multi-UAV bridge exited (N=$UAV_COUNT)"
