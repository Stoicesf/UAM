"""环境 Wrapper — 统一 obs/reward/done 接口，供 MAPPO 训练循环使用。"""

from __future__ import annotations

from typing import Any

import torch


class VMASWrapper:
    """轻量封装，后续可扩展为 gymnasium 兼容接口。"""

    def __init__(self, env, cfg: dict[str, Any]):
        self.env = env
        self.cfg = cfg
        self.n_agents = env.n_agents

    def reset(self):
        return self.env.reset()

    def step(self, actions: torch.Tensor):
        return self.env.step(actions)

    @property
    def device(self):
        return self.env.device
