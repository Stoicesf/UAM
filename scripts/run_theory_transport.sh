#!/usr/bin/env bash
# Launch AAS N-quad SITL + TransportHierarchicalController bridge.
# Run inside Ubuntu WSL:
#   bash /mnt/f/UAM/scripts/run_theory_transport.sh [target_x] [target_y]
#
# Dry physics-only (no Docker / AAS):
#   DRY_RUN=1 bash /mnt/f/UAM/scripts/run_theory_transport.sh
#
# Note: AAS has NO host colcon src/. Do not ros2 pkg create under aerial-autonomy-stack/.
# Bridge uses docker exec set_reposition — NOT host MAVROS / AttitudeTarget.

set -euo pipefail

UAM_ROOT="${UAM_ROOT:-/mnt/f/UAM}"
AAS_ROOT="${AAS_ROOT:-$HOME/aerial-autonomy-stack}"
UAV_COUNT="${UAV_COUNT:-4}"
ALTITUDE="${ALTITUDE:-12}"
HEADLESS="${HEADLESS:-false}"
DRY_RUN="${DRY_RUN:-0}"
TARGET_X="${1:-${TARGET_X:-5.0}}"
TARGET_Y="${2:-${TARGET_Y:-0.0}}"
STEPS="${STEPS:-80}"
WIND="${WIND:-0.0}"
NO_ACCEPT="${NO_ACCEPT:-0}"
OUTPUT="${OUTPUT:-}"
HYBRID="${HYBRID:-1}"
RISE="${RISE:-1}"
TRAJ="${TRAJ:-0}"
APF="${APF:-0}"

dk() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sudo docker "$@"
  fi
}

export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
export PYTHONPATH="$UAM_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BRIDGE=(
  python3 -u "$UAM_ROOT/ros_nodes/transport_theory_bridge.py"
  --uav_count "$UAV_COUNT"
  --altitude "$ALTITUDE"
  --target_x "$TARGET_X"
  --target_y "$TARGET_Y"
  --steps "$STEPS"
  --wind "$WIND"
)
if [[ "$HYBRID" == "1" || "$HYBRID" == "true" ]]; then
  BRIDGE+=(--hybrid)
else
  BRIDGE+=(--no_hybrid)
fi
if [[ "$RISE" == "1" || "$RISE" == "true" ]]; then
  BRIDGE+=(--rise)
else
  BRIDGE+=(--no_rise)
fi
if [[ "$TRAJ" == "1" || "$TRAJ" == "true" ]]; then
  BRIDGE+=(--traj)
fi
if [[ "$APF" == "1" || "$APF" == "true" ]]; then
  BRIDGE+=(--apf)
fi
if [[ "$NO_ACCEPT" == "1" || "$NO_ACCEPT" == "true" ]]; then
  BRIDGE+=(--no_accept)
fi
if [[ -n "$OUTPUT" ]]; then
  BRIDGE+=(--output "$OUTPUT")
fi

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "[dry_run] theory bridge host physics only — no AAS / Docker"
  "${BRIDGE[@]}" --dry_run 2>&1 | tee /tmp/theory_transport_dry.log
  exit "${PIPESTATUS[0]}"
fi

if [[ ! -d "$AAS_ROOT/tools_and_docs" ]]; then
  echo "AAS not found at $AAS_ROOT — clone aerial-autonomy-stack or use DRY_RUN=1"
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found — enable Docker Desktop WSL integration or use DRY_RUN=1"
  exit 1
fi
dk info >/dev/null 2>&1 || { sudo service docker start; sleep 2; }

cd "$AAS_ROOT/tools_and_docs"

dk ps -aq --filter name=inst0 | xargs -r dk rm -f || true
dk network rm aas-sim-network-inst0 aas-air-network-inst0 2>/dev/null || true

echo "[1/2] Starting NUM_QUADS=$UAV_COUNT HEADLESS=$HEADLESS (theory bridge)"
KEEPALIVE_SEC="${KEEPALIVE_SEC:-1200}"
( sleep "$KEEPALIVE_SEC"; printf 'x' ) | NUM_QUADS="$UAV_COUNT" HEADLESS="$HEADLESS" CAMERA=false LIDAR=false \
  WORLD=impalpable_greyness ./sim_run.sh 2>&1 | tee /tmp/theory_transport_sim.log &
SIM_PID=$!

echo "Waiting for aircraft containers…"
for _ in $(seq 1 90); do
  n=$(dk ps --format '{{.Names}}' | grep -c "aircraft-container-inst0_" || true)
  if [[ "$n" -ge "$UAV_COUNT" ]]; then
    break
  fi
  sleep 2
done
dk ps --format 'table {{.Names}}\t{{.Status}}'

echo "[1.5/2] GCS heartbeat feeder"
sleep 25
bash "$UAM_ROOT/scripts/aas_gcs_heartbeat.sh" "$UAV_COUNT"
sleep 5

echo "[2/2] Theory transport bridge"
"${BRIDGE[@]}" 2>&1 | tee /tmp/theory_transport_bridge.log
RC="${PIPESTATUS[0]}"

echo "Bridge finished (rc=$RC). Tearing down sim…"
dk ps -aq --filter name=inst0 | xargs -r dk rm -f || true
dk network rm aas-sim-network-inst0 aas-air-network-inst0 2>/dev/null || true
kill "$SIM_PID" 2>/dev/null || true
wait "$SIM_PID" 2>/dev/null || true
exit "$RC"
