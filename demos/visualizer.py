"""Matplotlib real-time visualizer for UAV swarm demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from demos.demo_runner import DemoFrame
from visualization.plot_utils import LEVEL_STYLE, ROLE_COLORS, role_color


class SwarmVisualizer:
    def __init__(
        self,
        boundary: float = 5.0,
        title: str = "UAM demo",
        *,
        show: bool = True,
        explain: bool = False,
    ):
        self.boundary = boundary
        self.explain = explain
        self.selected_agent: int | None = None
        self._timeline: dict[str, Any] | None = None
        self.fig, (self.ax_world, self.ax_dash) = plt.subplots(
            1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [2.2, 1.0]}
        )
        if self.fig.canvas.manager is not None:
            try:
                self.fig.canvas.manager.set_window_title(title)
            except Exception:
                pass
        self.ax_world.set_xlim(-boundary, boundary)
        self.ax_world.set_ylim(-boundary, boundary)
        self.ax_world.set_aspect("equal")
        self.ax_world.set_title("World")
        self.ax_dash.set_title("Dashboard")
        self.ax_dash.set_xlim(0, 1)
        self.ax_dash.set_ylim(0, 1)
        self.ax_dash.axis("off")

        self._scat = self.ax_world.scatter([], [], s=60, zorder=5, picker=bool(explain))
        self._tasks = self.ax_world.scatter([], [], marker="*", s=180, c="#f5c211", zorder=4, label="task")
        self._block = self.ax_world.scatter([], [], marker="s", s=50, c="#5e5c64", zorder=3, label="obstacle")
        self._payload = self.ax_world.scatter([], [], marker="o", s=220, c="#c64600", zorder=6, label="payload")
        self._target_cross: list[Any] = []
        self._cable_artists: list[Line2D] = []
        self._link_artists: list[Line2D] = []
        self._corr_quiver = None
        self._help_on = False
        self._status_artists: list[Any] = []
        self._bar_artists: list[Any] = []
        self._trail_line: Line2D | None = None

        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=name)
            for name, c in (
                ("SCOUT", ROLE_COLORS[0]),
                ("EXECUTOR", ROLE_COLORS[1]),
                ("RELAY", ROLE_COLORS[2]),
                ("GUARDIAN", ROLE_COLORS[3]),
                ("LOGISTICS", ROLE_COLORS[4]),
            )
        ]
        handles += [
            Line2D([0], [0], color=LEVEL_STYLE[1]["color"], linestyle="--", lw=1, label="L1"),
            Line2D([0], [0], color=LEVEL_STYLE[2]["color"], linestyle="-", lw=1.5, label="L2"),
            Line2D([0], [0], color=LEVEL_STYLE[3]["color"], linestyle="-", lw=2.5, label="L3"),
        ]
        self.ax_world.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.8)

        self.fig.tight_layout()
        if show:
            plt.ion()
            self.fig.show()

    def set_timeline(self, timeline: dict[str, Any] | None) -> None:
        self._timeline = timeline

    def on_pick(self, event) -> int | None:
        if event.artist is not self._scat or not len(event.ind):
            return None
        self.selected_agent = int(event.ind[0])
        return self.selected_agent

    def toggle_help(self) -> None:
        self._help_on = not self._help_on

    def update(self, frame: DemoFrame, scene_name: str = "") -> None:
        pos = frame.pos.cpu().numpy()
        alive = frame.alive.cpu().numpy().astype(bool)
        roles = frame.roles.cpu().numpy().astype(int)
        colors = [role_color(r) if a else "#9a9996" for r, a in zip(roles, alive)]
        if frame.uav_sizes:
            sizes = np.asarray(frame.uav_sizes, dtype=float)
            sizes = np.where(alive, sizes, sizes * 0.35)
        else:
            sizes = np.where(alive, 70, 25)
        if self.selected_agent is not None and 0 <= self.selected_agent < len(sizes):
            sizes[self.selected_agent] = max(float(sizes[self.selected_agent]), 140)
        self._scat.set_offsets(pos)
        self._scat.set_color(colors)
        self._scat.set_sizes(sizes)

        tasks = frame.tasks.cpu().numpy()
        done = frame.task_done.cpu().numpy().astype(bool)
        if tasks.size:
            self._tasks.set_offsets(tasks[:, :2])
            self._tasks.set_color(["#9a9996" if d else "#f5c211" for d in done])
        else:
            self._tasks.set_offsets(np.zeros((0, 2)))

        blockers = frame.blockers.cpu().numpy()
        self._block.set_offsets(blockers if blockers.size else np.zeros((0, 2)))

        for ln in self._link_artists:
            ln.remove()
        self._link_artists.clear()
        for ln in self._cable_artists:
            ln.remove()
        self._cable_artists.clear()
        for L in frame.links:
            if L.get("to_payload") and frame.payload_pos is not None:
                pp = frame.payload_pos.cpu().numpy().reshape(-1)
                i = int(L["src"])
                (ln,) = self.ax_world.plot(
                    [pos[i, 0], pp[0]],
                    [pos[i, 1], pp[1]],
                    color="#c64600",
                    linestyle="-",
                    lw=1.2,
                    alpha=0.55,
                    zorder=2,
                )
                self._cable_artists.append(ln)
                continue
            st = LEVEL_STYLE.get(int(L["level"]), LEVEL_STYLE[1])
            i, j = int(L["src"]), int(L["dst"])
            dist = float(L.get("dist", 1.0))
            alpha = max(0.15, float(st["alpha"]) * (1.0 / (1.0 + 0.35 * dist)))
            (ln,) = self.ax_world.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                linestyle=st["linestyle"],
                color=st["color"],
                linewidth=st["linewidth"],
                alpha=alpha,
                zorder=2,
            )
            self._link_artists.append(ln)

        if frame.payload_pos is not None:
            pp = frame.payload_pos.cpu().numpy().reshape(1, 2)
            self._payload.set_offsets(pp)
        else:
            self._payload.set_offsets(np.zeros((0, 2)))
        for art in self._target_cross:
            art.remove()
        self._target_cross.clear()
        if frame.target_pos is not None:
            tp = frame.target_pos.cpu().numpy().reshape(-1)
            (h,) = self.ax_world.plot(
                [tp[0] - 0.4, tp[0] + 0.4], [tp[1], tp[1]], color="#2ec27e", lw=2, zorder=4
            )
            (v,) = self.ax_world.plot(
                [tp[0], tp[0]], [tp[1] - 0.4, tp[1] + 0.4], color="#2ec27e", lw=2, zorder=4
            )
            self._target_cross.extend([h, v])

        if self._corr_quiver is not None:
            self._corr_quiver.remove()
            self._corr_quiver = None
        if frame.correction is not None:
            c = frame.correction.cpu().numpy()
            mag = np.linalg.norm(c, axis=1)
            mask = mag > 1e-3
            if mask.any():
                self._corr_quiver = self.ax_world.quiver(
                    pos[mask, 0],
                    pos[mask, 1],
                    c[mask, 0],
                    c[mask, 1],
                    color="#e01b24",
                    scale=8,
                    width=0.004,
                    zorder=6,
                )

        if self._trail_line is not None:
            self._trail_line.remove()
            self._trail_line = None
        if self._timeline and self._timeline.get("pos"):
            trail = np.asarray(self._timeline["pos"], dtype=float)
            if trail.size:
                (self._trail_line,) = self.ax_world.plot(
                    trail[:, 0], trail[:, 1], color="#f6d32d", lw=2, alpha=0.9, zorder=7
                )

        if self.explain and self._timeline is not None:
            self._draw_explain(frame, scene_name)
        else:
            self._draw_dashboard(frame, scene_name)
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            self.fig.canvas.draw()

    def _draw_explain(self, frame: DemoFrame, scene_name: str) -> None:
        for a in self._status_artists + self._bar_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._status_artists.clear()
        self._bar_artists.clear()
        ax = self.ax_dash
        tl = self._timeline or {}
        aid = tl.get("agent_id", self.selected_agent)
        roles = tl.get("roles", [])
        levels = tl.get("levels", [])
        gains = tl.get("gains", [])
        costs = tl.get("costs", [])
        steps = tl.get("steps", [])
        lines = [
            f"EXPLAIN UAV #{aid}",
            f"Scene: {scene_name}",
            f"Global step: {frame.step}",
            f"Last gain: {gains[-1]:.3f}" if gains else "Last gain: -",
            f"Last cost: {costs[-1]:.3f}" if costs else "Last cost: -",
            f"Roles: {roles}",
            f"Levels: {levels}",
        ]
        switches = [i for i in range(1, len(roles)) if roles[i] != roles[i - 1]]
        l3_steps = [steps[i] for i, lv in enumerate(levels) if lv == 3]
        lines.append(f"Role switches @ idx: {switches or 'none'}")
        lines.append(f"L3 steps: {l3_steps or 'none'}")
        if self._help_on:
            lines += ["", "Click UAV to inspect"]
        y0 = 0.95
        for i, t in enumerate(lines):
            self._status_artists.append(ax.text(0.04, y0 - i * 0.05, t, fontsize=8, family="monospace", va="top"))
        if levels:
            from collections import Counter

            c = Counter(levels)
            tot = sum(c.values()) + 1e-8
            base_y = 0.18
            for i, lv in enumerate((1, 2, 3)):
                sh = c.get(lv, 0) / tot
                y = base_y - i * 0.05
                self._bar_artists.append(
                    ax.add_patch(plt.Rectangle((0.05, y), 0.9 * sh, 0.035, color=LEVEL_STYLE[lv]["color"]))
                )
                self._bar_artists.append(ax.text(0.05, y + 0.038, f"L{lv} {sh*100:.0f}%", fontsize=8))

    def _draw_dashboard(self, frame: DemoFrame, scene_name: str) -> None:
        for a in self._status_artists + self._bar_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._status_artists.clear()
        self._bar_artists.clear()
        ax = self.ax_dash
        bw_pct = 100.0 * frame.bandwidth_hz / max(frame.bandwidth_max, 1.0)
        lines = [
            f"Scene: {scene_name or '-'}",
            f"Step: {frame.step}",
            f"Reward: {frame.reward:.2f}",
            f"Coverage: {frame.coverage:.2f}",
            f"Bandwidth: {bw_pct:.0f}%",
            f"Bytes step: {frame.bytes_step:.0f}",
            f"Bytes cum: {frame.bytes_cum:.0f}",
            f"Alive: {int(frame.alive.sum())}/{frame.alive.numel()}",
        ]
        if frame.payload_pos is not None:
            lines.insert(4, f"Payload dist: {frame.payload_distance:.2f}")
        if self._help_on:
            lines += ["", "Keys:", "Space pause", "R reset", "H help", "Q quit"]
            if self.explain:
                lines.append("Click UAV explain")
        y0 = 0.95
        for i, t in enumerate(lines):
            self._status_artists.append(ax.text(0.05, y0 - i * 0.055, t, fontsize=9, family="monospace", va="top"))

        counts = frame.level_counts
        tot = sum(counts.get(k, 0) for k in (1, 2, 3)) + 1e-8
        shares = [counts.get(1, 0) / tot, counts.get(2, 0) / tot, counts.get(3, 0) / tot]
        base_y = 0.22
        for i, (lab, sh) in enumerate(zip(("L1", "L2", "L3"), shares)):
            y = base_y - i * 0.06
            lv = i + 1
            self._bar_artists.append(
                ax.add_patch(plt.Rectangle((0.05, y), 0.9 * sh, 0.04, color=LEVEL_STYLE[lv]["color"], alpha=0.85))
            )
            self._bar_artists.append(ax.text(0.05, y + 0.045, f"{lab} {sh*100:.0f}%", fontsize=8, va="bottom"))

    def save_explain_snapshot(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=120)

    def close(self) -> None:
        plt.close(self.fig)
