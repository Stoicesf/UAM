"""Piecewise minimum-snap / smooth trajectory (NKU-style flat output, 2D).

Uses scipy.optimize when available; otherwise closed-form piecewise quintic (C2).
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import minimize

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


class MinimumSnapTrajectory:
    """
    Flat outputs: x_L(t), y_L(t).
    Default order=5 (quintic) is enough for continuous acceleration;
    with scipy, fits higher-order polys minimizing snap-like cost.
    """

    def __init__(
        self,
        waypoints: list[list[float]] | np.ndarray,
        total_time: float,
        order: int = 5,
        n_segments: int | None = None,
    ):
        wps = np.asarray(waypoints, dtype=np.float64)
        if wps.ndim != 2 or wps.shape[1] < 2:
            raise ValueError("waypoints must be [M, 2+]")
        self.waypoints = wps[:, :2]
        self.T = float(total_time)
        self.order = int(order)
        self.n_segments = int(n_segments) if n_segments else max(1, len(self.waypoints) - 1)
        self.seg_T = self.T / self.n_segments
        self.coeffs: np.ndarray | None = None  # [seg, order+1, 2]

    def generate(self) -> np.ndarray:
        if _HAS_SCIPY and self.order >= 5:
            self.coeffs = self._fit_scipy()
        else:
            self.coeffs = self._fit_quintic()
        return self.coeffs

    def _fit_quintic(self) -> np.ndarray:
        """Closed-form rest-to-rest quintic per segment (acc continuous at knots ≈0)."""
        order = 5
        coeffs = np.zeros((self.n_segments, order + 1, 2))
        for seg in range(self.n_segments):
            p0 = self.waypoints[min(seg, len(self.waypoints) - 1)]
            p1 = self.waypoints[min(seg + 1, len(self.waypoints) - 1)]
            T = self.seg_T
            # p(0)=p0, p(T)=p1, p'=p''=0 at ends
            # p = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
            for d in range(2):
                a0 = p0[d]
                a1 = 0.0
                a2 = 0.0
                # solve for a3,a4,a5 from p(T), p'(T), p''(T)=0
                # p(T)=a0+a3 T^3+a4 T^4+a5 T^5 = p1
                # p'=3a3 T^2+4a4 T^3+5a5 T^4 = 0
                # p''=6a3 T+12a4 T^2+20a5 T^3 = 0
                A = np.array(
                    [
                        [T**3, T**4, T**5],
                        [3 * T**2, 4 * T**3, 5 * T**4],
                        [6 * T, 12 * T**2, 20 * T**3],
                    ],
                    dtype=np.float64,
                )
                b = np.array([p1[d] - a0, 0.0, 0.0], dtype=np.float64)
                a345 = np.linalg.solve(A, b)
                coeffs[seg, 0, d] = a0
                coeffs[seg, 1, d] = a1
                coeffs[seg, 2, d] = a2
                coeffs[seg, 3:, d] = a345
        self.order = order
        return coeffs

    def _fit_scipy(self) -> np.ndarray:
        """Per-axis / per-segment poly fit minimizing sum of high-order coeffs^2."""
        n = self.order + 1
        coeffs = np.zeros((self.n_segments, n, 2))
        for seg in range(self.n_segments):
            p0 = self.waypoints[min(seg, len(self.waypoints) - 1)]
            p1 = self.waypoints[min(seg + 1, len(self.waypoints) - 1)]
            T = self.seg_T
            for d in range(2):

                def cost(c: np.ndarray) -> float:
                    # snap ~ 4th derivative energy ≈ weight on coeffs k>=4
                    w = np.array([(k * (k - 1) * (k - 2) * (k - 3)) ** 2 for k in range(n)])
                    return float(np.dot(w, c**2))

                cons = [
                    {"type": "eq", "fun": lambda c, p=p0[d]: float(c[0] - p)},
                    {
                        "type": "eq",
                        "fun": lambda c, p=p1[d], tt=T: float(
                            sum(c[k] * (tt**k) for k in range(n)) - p
                        ),
                    },
                    {"type": "eq", "fun": lambda c: float(c[1] if n > 1 else 0.0)},
                    {
                        "type": "eq",
                        "fun": lambda c, tt=T: float(
                            sum(k * c[k] * (tt ** (k - 1)) for k in range(1, n))
                        ),
                    },
                ]
                c0 = np.zeros(n)
                c0[0] = p0[d]
                if n > 1:
                    c0[1] = (p1[d] - p0[d]) / max(T, 1e-6)
                res = minimize(cost, c0, constraints=cons, method="SLSQP")
                coeffs[seg, :, d] = res.x if res.success else c0
        return coeffs

    def compute(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.coeffs is None:
            self.generate()
        assert self.coeffs is not None
        t = float(np.clip(t, 0.0, self.T))
        seg = int(t / self.seg_T) if self.seg_T > 0 else 0
        seg = min(seg, self.n_segments - 1)
        tau = t - seg * self.seg_T
        pos = np.zeros(2)
        vel = np.zeros(2)
        acc = np.zeros(2)
        c = self.coeffs[seg]
        n = c.shape[0]
        for d in range(2):
            pos[d] = sum(c[k, d] * (tau**k) for k in range(n))
            vel[d] = sum(k * c[k, d] * (tau ** (k - 1)) for k in range(1, n))
            acc[d] = sum(
                k * (k - 1) * c[k, d] * (tau ** (k - 2)) for k in range(2, n)
            )
        return pos, vel, acc


def self_check() -> None:
    wps = [[0.0, 0.0], [2.5, 2.5], [5.0, 5.0]]
    traj = MinimumSnapTrajectory(wps, total_time=10.0, order=5)
    traj.generate()
    p0, v0, a0 = traj.compute(0.0)
    p1, v1, a1 = traj.compute(5.0)
    pT, vT, aT = traj.compute(10.0)
    assert np.allclose(p0, [0.0, 0.0], atol=1e-3)
    assert np.allclose(pT, [5.0, 5.0], atol=1e-2)
    # mid should be near second waypoint neighborhood
    assert float(np.linalg.norm(p1 - np.array([2.5, 2.5]))) < 1.5
    # acceleration finite / no nan
    assert np.isfinite(a0).all() and np.isfinite(a1).all() and np.isfinite(aT).all()
    print("minimum_snap: OK")


if __name__ == "__main__":
    self_check()
