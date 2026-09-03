"""Onboard compute quota simulator (Jetson Nano / Orin NX / Orin AGX).

inference_time = alpha * model_params + beta * obs_dim   (ms), scaled by 1/tops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class JetsonProfile:
    name: str
    tops: float
    mem_gb: float
    tier: str  # high | mid | low


PROFILES: dict[str, JetsonProfile] = {
    "orin_agx": JetsonProfile("Jetson Orin AGX", tops=275.0, mem_gb=64.0, tier="high"),
    "orin_nx": JetsonProfile("Jetson Orin NX", tops=100.0, mem_gb=16.0, tier="mid"),
    "nano": JetsonProfile("Jetson Nano", tops=0.5, mem_gb=4.0, tier="low"),
}

# aliases matching plan wording
TIER_PROFILES = {
    "high": PROFILES["orin_agx"],
    "mid": PROFILES["orin_nx"],
    "low": PROFILES["nano"],
}


def inference_time_ms(
    model_params: float,
    obs_dim: int,
    tops_quota: float,
    *,
    alpha: float = 1e-5,
    beta: float = 0.05,
    ref_tops: float = 100.0,
) -> float:
    """Plan model: alpha * params + beta * obs_dim, slowed on smaller TOPS."""
    base = alpha * model_params + beta * float(obs_dim)
    scale = ref_tops / max(tops_quota, 1e-6)
    return float(base * scale)


def estimate_latency_ms(
    model_flops: float,
    input_dim: int,
    tops_quota: float,
    *,
    overhead_ms: float = 2.0,
    alpha: float = 1e-5,
    beta: float = 0.05,
) -> float:
    """Backward-compatible wrapper; prefers alpha/beta param model when flops≈params."""
    return overhead_ms + inference_time_ms(
        model_flops, input_dim, tops_quota, alpha=alpha, beta=beta
    )


def flops_mlp(in_dim: int, hidden: int, out_dim: int) -> float:
    return float(2 * (in_dim * hidden + hidden * out_dim))


def inject_compute_delay(latency_ms: float, *, real_sleep: bool = False, sleep_scale: float = 0.1) -> float:
    """Accumulate / optionally sleep. Plan: sleep(inference_time/10) → sleep_scale=0.1.

    ponytail: default real_sleep=False for vectorized experiments (virtual delay only).
    """
    if real_sleep and latency_ms > 0:
        time.sleep(min(latency_ms * sleep_scale / 1000.0, 0.05))
    return latency_ms


class ComputeBudgetEnv:
    """Wraps decision steps with profile-dependent latency bookkeeping."""

    def __init__(self, profile_key: str = "mid", model_params: float = 5e5, obs_dim: int = 64):
        self.profile = TIER_PROFILES.get(profile_key, PROFILES.get(profile_key, TIER_PROFILES["mid"]))
        self.model_params = model_params
        self.obs_dim = obs_dim
        self.total_delay_ms = 0.0
        self.steps = 0

    def before_decision(self, real_sleep: bool = False) -> float:
        lat = inference_time_ms(self.model_params, self.obs_dim, self.profile.tops)
        inject_compute_delay(lat, real_sleep=real_sleep)
        self.total_delay_ms += lat
        self.steps += 1
        return lat

    @property
    def mean_delay_ms(self) -> float:
        return self.total_delay_ms / max(self.steps, 1)


def self_check() -> None:
    assert set(PROFILES) >= {"nano", "orin_nx", "orin_agx"}
    t_hi = inference_time_ms(5e5, 64, TIER_PROFILES["high"].tops)
    t_lo = inference_time_ms(5e5, 64, TIER_PROFILES["low"].tops)
    assert t_lo > t_hi
    env = ComputeBudgetEnv("low")
    env.before_decision(real_sleep=False)
    assert env.mean_delay_ms > 0
    print(f"resource_simulator: OK (high={t_hi:.3f}ms low={t_lo:.3f}ms)")


if __name__ == "__main__":
    self_check()
