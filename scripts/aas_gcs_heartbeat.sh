#!/usr/bin/env bash
# Keep PX4 preflight happy by feeding GCS heartbeats into each SITL mavlink port.
# AAS PX4 instance i listens UDP 14580+i and talks to remote 14540+i (inside sim container).
set -euo pipefail
N="${1:-4}"
DK=(docker)
if ! docker info >/dev/null 2>&1; then
  DK=(sudo docker)
fi
"${DK[@]}" exec -d simulation-container-inst0 bash -lc "
python3 - <<'PY'
import time
from pymavlink import mavutil
n = $N
conns = []
for i in range(n):
    # PX4 SITL GCS remote port (classic) + onboard twin
    for port in (14540 + i, 14580 + i):
        try:
            conns.append(mavutil.mavlink_connection(f'udpout:127.0.0.1:{port}'))
        except Exception as e:
            print('skip', port, e)
print('gcs heartbeat feeder ports', [14540+i for i in range(n)] + [14580+i for i in range(n)], flush=True)
while True:
    for m in conns:
        m.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0)
    time.sleep(0.5)
PY
"
echo "GCS heartbeat feeder started in simulation-container-inst0"
