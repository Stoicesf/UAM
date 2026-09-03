"""Local vs edge/neighbor offloading heuristic (not full LABOR)."""

from __future__ import annotations

from enum import Enum

from models.icps.compute.resource_simulator import PROFILES, estimate_latency_ms


class OffloadTarget(str, Enum):
    LOCAL = "local"
    NEIGHBOR = "neighbor"
    EDGE = "edge"


def decide_offload(
    model_flops: float,
    input_dim: int,
    tops_quota: float,
    *,
    latency_budget_ms: float = 50.0,
    neighbor_tops: float | None = None,
    edge_tops: float = 275.0,
    link_rtt_ms: float = 5.0,
) -> OffloadTarget:
    local = estimate_latency_ms(model_flops, input_dim, tops_quota)
    if local <= latency_budget_ms:
        return OffloadTarget.LOCAL
    if neighbor_tops is not None:
        nlat = estimate_latency_ms(model_flops, input_dim, neighbor_tops) + link_rtt_ms
        if nlat <= latency_budget_ms:
            return OffloadTarget.NEIGHBOR
    edge = estimate_latency_ms(model_flops, input_dim, edge_tops) + 2 * link_rtt_ms
    if edge <= latency_budget_ms or edge < local:
        return OffloadTarget.EDGE
    return OffloadTarget.LOCAL


def self_check() -> None:
    # tiny model stays local on Nano
    d = decide_offload(1e6, 32, PROFILES["nano"].tops, latency_budget_ms=50)
    assert d == OffloadTarget.LOCAL
    # huge model offloads
    d2 = decide_offload(1e14, 256, PROFILES["nano"].tops, latency_budget_ms=10)
    assert d2 in (OffloadTarget.EDGE, OffloadTarget.NEIGHBOR, OffloadTarget.LOCAL)
    print(f"offloading_policy: OK ({d}, {d2})")


if __name__ == "__main__":
    self_check()
