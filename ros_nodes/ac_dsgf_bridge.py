#!/usr/bin/env python3
"""ROS2 bridge: CompleteController → MAVROS velocity setpoints (Gazebo/PX4).

  python ros_nodes/ac_dsgf_bridge.py --ros-args \\
    -p uav_count:=4 -p shield_type:=cbf -p max_vel:=2.0

# ponytail: Lissajous goals avoid pile-up; offboard arming left to AAS launch.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_alt = Path(os.path.expanduser("~/ac-dsgf-swarm"))
if _alt.exists() and str(_alt) not in sys.path:
    sys.path.insert(0, str(_alt))

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from rclpy.node import Node
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "rclpy not found — run this node inside a sourced ROS2 workspace.\n"
        f"ImportError: {e}"
    ) from e

from models.complete_controller import CompleteController


class ACDSGFBridge(Node):
    def __init__(self) -> None:
        super().__init__("ac_dsgf_bridge")

        self.declare_parameter("model_path", str(_ROOT / "experiment_results" / "semantic" / "semantic_encoder.pt"))
        self.declare_parameter("uav_count", 4)
        self.declare_parameter("namespace_prefix", "uav")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("shield_type", "cbf")
        self.declare_parameter("obs_dim", 48)
        self.declare_parameter("max_vel", 2.0)

        model_path = self.get_parameter("model_path").value
        self.uav_count = int(self.get_parameter("uav_count").value)
        self.ns_prefix = str(self.get_parameter("namespace_prefix").value)
        self.rate = float(self.get_parameter("control_rate_hz").value)
        shield_type = str(self.get_parameter("shield_type").value)
        obs_dim = int(self.get_parameter("obs_dim").value)
        self.max_vel = float(self.get_parameter("max_vel").value)
        n_tasks = self.uav_count  # one Lissajous waypoint per UAV

        self.controller = CompleteController(
            self.uav_count,
            n_tasks,
            obs_dim,
            n_roles=3,
            use_semantic=True,
            use_icps_resource=False,
            use_dice=True,
            shield_type=shield_type,
        )
        self.controller.eval()
        ckpt = Path(model_path)
        if ckpt.exists():
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "state_dict" in payload:
                if payload.get("obs_dim") == obs_dim:
                    self.controller.encoder.load_state_dict(payload["state_dict"])
                    self.get_logger().info(f"Loaded encoder from {ckpt}")
                else:
                    self.get_logger().warn(
                        f"Skip encoder ckpt obs_dim={payload.get('obs_dim')} != {obs_dim}"
                    )
            else:
                self.get_logger().warn(f"Unrecognized checkpoint format: {ckpt}")
        else:
            self.get_logger().warn(f"No checkpoint at {ckpt}; running untrained controller")

        self.pos = torch.zeros(self.uav_count, 2)
        self.vel = torch.zeros(self.uav_count, 2)
        self.roles = torch.zeros(self.uav_count, dtype=torch.long)
        self.alive = torch.ones(self.uav_count, dtype=torch.bool)
        self.tasks = torch.zeros(n_tasks, 4)
        self.tasks[:, 2] = 1.0
        self.task_done = torch.zeros(n_tasks, dtype=torch.bool)
        self.assignment = torch.arange(self.uav_count, dtype=torch.long)  # i → task i
        self._pose_ok = [False] * self.uav_count
        self.step_count = 0

        self.pose_subscribers = []
        self.vel_subscribers = []
        for i in range(self.uav_count):
            ns = f"/{self.ns_prefix}{i + 1}"
            pose_topic = f"{ns}/mavros/local_position/pose"
            vel_topic = f"{ns}/mavros/local_position/velocity_local"
            self.pose_subscribers.append(
                self.create_subscription(PoseStamped, pose_topic, lambda msg, idx=i: self._on_pose(msg, idx), 10)
            )
            self.vel_subscribers.append(
                self.create_subscription(Twist, vel_topic, lambda msg, idx=i: self._on_vel(msg, idx), 10)
            )
            self.get_logger().info(f"Subscribed {pose_topic} + {vel_topic}")

        self.cmd_publishers = []
        for i in range(self.uav_count):
            topic = f"/{self.ns_prefix}{i + 1}/mavros/setpoint_velocity/cmd_vel_unstamped"
            self.cmd_publishers.append(self.create_publisher(Twist, topic, 10))
            self.get_logger().info(f"Publishing {topic}")

        self.create_timer(1.0 / max(self.rate, 1.0), self.control_loop)
        self.get_logger().info(
            f"Bridge @ {self.rate} Hz, N={self.uav_count}, shield={shield_type}, max_vel={self.max_vel}"
        )

    def get_dynamic_tasks(self) -> torch.Tensor:
        """Per-UAV Lissajous waypoints (phase-staggered to reduce mid-air pile-up)."""
        t = self.step_count * 0.02
        rows = []
        for i in range(self.uav_count):
            phase = 2 * math.pi * i / max(self.uav_count, 1)
            r = 3.0 + 1.0 * math.sin(t * 0.2 + phase)
            x = r * math.cos(t * 0.3 + phase)
            y = r * math.sin(t * 0.4 + phase)
            rows.append([x, y, 1.0, 1e6])
        return torch.tensor(rows, dtype=torch.float32)

    def _on_pose(self, msg: PoseStamped, idx: int) -> None:
        self.pos[idx, 0] = float(msg.pose.position.x)
        self.pos[idx, 1] = float(msg.pose.position.y)
        self._pose_ok[idx] = True

    def _on_vel(self, msg: Twist, idx: int) -> None:
        self.vel[idx, 0] = float(msg.linear.x)
        self.vel[idx, 1] = float(msg.linear.y)

    def _fake_obs(self) -> torch.Tensor:
        n = self.uav_count
        d = int(self.controller.encoder.enc[0].in_features)
        obs = torch.zeros(n, d)
        obs[:, 0:2] = self.pos
        obs[:, 2:4] = self.vel
        return obs

    def control_loop(self) -> None:
        if not all(self._pose_ok):
            self.get_logger().warn("Waiting for all UAV poses…", throttle_duration_sec=2.0)
            return
        if torch.isnan(self.pos).any() or torch.isnan(self.vel).any():
            self.get_logger().warn("NaN in state, skip")
            return

        self.tasks = self.get_dynamic_tasks()
        # blend controller output with direct waypoint chase (physics has lag)
        goals = self.tasks[:, :2]
        to_goal = (goals - self.pos).clamp(-1, 1)

        obs = self._fake_obs()
        with torch.no_grad():
            out = self.controller.step(
                obs,
                self.pos,
                self.vel,
                self.tasks,
                self.roles,
                self.alive,
                self.task_done,
                self.assignment,
                snr_db=20.0,
            )
        self.roles = out.roles
        dxdy = (0.35 * out.dxdy + 0.65 * to_goal).clamp(-1, 1)
        cmd = dxdy * self.max_vel

        dists = (goals - self.pos).norm(dim=-1)
        min_pair = float("inf")
        for i in range(self.uav_count):
            for j in range(i + 1, self.uav_count):
                min_pair = min(min_pair, float((self.pos[i] - self.pos[j]).norm()))
        if self.step_count % int(max(self.rate, 1)) == 0:
            d_str = ",".join(f"{float(d):.2f}" for d in dists)
            self.get_logger().info(f"Dist→goal=[{d_str}] min_sep={min_pair:.2f}m")

        for i in range(self.uav_count):
            twist = Twist()
            twist.linear.x = float(cmd[i, 0])
            twist.linear.y = float(cmd[i, 1])
            twist.linear.z = 0.0
            self.cmd_publishers[i].publish(twist)

        self.step_count += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ACDSGFBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down bridge")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
