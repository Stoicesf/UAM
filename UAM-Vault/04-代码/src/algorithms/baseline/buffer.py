"""Rollout Buffer — 复用 TorchRL ReplayBuffer。"""

from __future__ import annotations

from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage


def make_replay_buffer(frames_per_batch: int, minibatch_size: int, device) -> ReplayBuffer:
    """On-policy replay buffer，每轮 rollout 后 refill。"""
    return ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )
