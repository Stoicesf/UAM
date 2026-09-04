#!/usr/bin/env bash
# Launch AAS 4-quad SITL then host-side AC-DSGF command bridge.
# Run inside Ubuntu-24.04 WSL:
#   bash /mnt/f/UAM/scripts/run_aas_bridge.sh
#
# Note: AAS has NO host colcon src/. Do not ros2 pkg create under aerial-autonomy-stack/.

set -euo pipefail

UAM_ROOT="${UAM_ROOT:-/mnt/f/UAM}"
AAS_ROOT="${AAS_ROOT:-$HOME/aerial-autonomy-stack}"
UAV_COUNT="${UAV_COUNT:-4}"
ALTITUDE="${ALTITUDE:-12}"
HEADLESS="${HEADLESS:-true}"

export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"

cd "$AAS_ROOT/tools_and_docs"

# cleanup leftover instance 0
docker ps -aq --filter name=inst0 | xargs -r docker rm -f || true
docker network rm aas-sim-network-inst0 aas-air-network-inst0 2>/dev/null || true

echo "[1/2] Starting NUM_QUADS=$UAV_COUNT HEADLESS=$HEADLESS"
( sleep 240; printf 'x' ) | NUM_QUADS="$UAV_COUNT" HEADLESS="$HEADLESS" CAMERA=false LIDAR=false \
  WORLD=impalpable_greyness ./sim_run.sh 2>&1 | tee /tmp/aas_bridge_sim.log &
SIM_PID=$!

echo "Waiting for aircraft containers…"
for _ in $(seq 1 90); do
  n=$(docker ps --format '{{.Names}}' | grep -c "aircraft-container-inst0_" || true)
  if [[ "$n" -ge "$UAV_COUNT" ]]; then
    break
  fi
  sleep 2
done
docker ps --format 'table {{.Names}}\t{{.Status}}'

echo "[1.5/2] GCS heartbeat feeder (PX4 preflight requires GCS link)"
# wait for sim PX4 to bind mavlink ports
sleep 25
bash "$UAM_ROOT/scripts/aas_gcs_heartbeat.sh" "$UAV_COUNT"
sleep 5

echo "[2/2] Host bridge (takeoff + reposition; numpy default, --brain for torch)"
export PYTHONPATH="$UAM_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
python3 -u "$UAM_ROOT/ros_nodes/aas_cmd_bridge.py" \
  --uav_count "$UAV_COUNT" \
  --altitude "$ALTITUDE" \
  --rate_hz 0.4 \
  --steps 30 \
  --shield cbf

echo "Bridge finished. Press a key in the sim_run terminal (or wait) to tear down."
wait "$SIM_PID" || true
