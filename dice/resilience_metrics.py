"""Resilience metrics."""

from __future__ import annotations


class ResilienceMetrics:
    @staticmethod
    def task_completion_rate(completed: int, total: int) -> float:
        return completed / max(total, 1)

    @staticmethod
    def recovery_time(failure_step: int, recovery_step: int) -> int:
        return max(0, recovery_step - failure_step)

    @staticmethod
    def performance_degradation(baseline: float, post: float) -> float:
        return (baseline - post) / max(abs(baseline), 1e-8)

    @staticmethod
    def cascade_threshold(crash_ratios: list[float], crashed: list[bool]) -> float:
        """Lowest failure ratio that caused crash; 1.0 if never."""
        hits = [r for r, c in zip(crash_ratios, crashed) if c]
        return min(hits) if hits else 1.0


def self_check() -> None:
    assert abs(ResilienceMetrics.task_completion_rate(7, 10) - 0.7) < 1e-9
    assert ResilienceMetrics.recovery_time(50, 80) == 30
    print("resilience_metrics: OK")


if __name__ == "__main__":
    self_check()
