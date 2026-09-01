"""UAV dataset engine — single source of truth for c_t teacher."""

from secdo.datasets.uav.channel import ChannelConfig, observe_channel
from secdo.datasets.uav.mobility import FastDrift, RandomWaypoint, SlowDrift, get_mobility
from secdo.datasets.uav.teacher import capacity_teacher, pack_state

__all__ = [
    "ChannelConfig",
    "observe_channel",
    "capacity_teacher",
    "pack_state",
    "SlowDrift",
    "FastDrift",
    "RandomWaypoint",
    "get_mobility",
]
