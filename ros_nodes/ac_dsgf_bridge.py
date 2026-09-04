#!/usr/bin/env python3
"""LEGACY MAVROS-oriented bridge (not for current AAS Docker SITL).

AAS uses per-drone ROS_DOMAIN_ID + PX4 uXRCE-DDS + set_reposition / offboard
actions inside docker networks — not /uavN/mavros/... on the host.

Use instead:
  python3 ros_nodes/aas_cmd_bridge.py --uav_count 4
  # or
  bash scripts/run_aas_bridge.sh
"""

raise SystemExit(
    "ac_dsgf_bridge.py is the old MAVROS sketch.\n"
    "For AAS Docker SITL run: python3 ros_nodes/aas_cmd_bridge.py --uav_count 4\n"
    "See scripts/run_aas_bridge.sh"
)
