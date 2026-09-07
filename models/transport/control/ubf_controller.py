"""UBF load position controller (2D, HIT-style simplified)."""

from __future__ import annotations

import torch
import torch.nn as nn


class UBFLoadController(nn.Module):
    """Universal Barrier Function position loop for payload tracking in 2D."""

    def __init__(
        self,
        K1L: float = 2.0,
        K2L: float = 2.0,
        eta1L: float = 0.25,
        delta_dL: float = 1.0,
        epsilon: float = 0.01,
    ):
        super().__init__()
        self.K1L = float(K1L)
        self.K2L = float(K2L)
        self.eta1L = float(eta1L)
        self.delta_dL = float(delta_dL)
        self.epsilon = float(epsilon)

    def forward(
        self,
        x_L: torch.Tensor,
        x_dL: torch.Tensor,
        v_L: torch.Tensor,
        v_dL: torch.Tensor,
        m_L: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (F_d [2], eta_eL scalar, z_2L [2]).
        """
        d_eL = torch.norm(x_L - x_dL).clamp(min=1e-8)
        # stay inside barrier ball
        d_safe = torch.clamp(d_eL, max=self.delta_dL - self.epsilon)
        E_L = (x_L - x_dL) / d_eL

        denom = (self.delta_dL - d_safe + self.epsilon) ** 2
        eta_eL = (self.delta_dL * d_safe) / (self.delta_dL - d_safe + self.epsilon)
        d_eta_d_delta = (self.delta_dL**2) / denom
        d_eta_d_d = -self.delta_dL * d_safe / denom

        # stabilizing function → desired velocity (v_dL feedforward)
        alpha_pL = E_L / d_eta_d_delta * (-d_eta_d_d * 0.0 - self.K1L * eta_eL) + v_dL
        z_2L = v_L - alpha_pL

        F_d = (
            m_L * alpha_pL
            - self.K2L * z_2L
            - eta_eL * d_eta_d_delta * E_L
            - self._adapt_term(z_2L)
        )
        return F_d, eta_eL, z_2L

    def _adapt_term(self, z: torch.Tensor) -> torch.Tensor:
        norm_z = torch.norm(z)
        if float(norm_z) > 1e-6:
            return self.eta1L * z / norm_z
        return torch.zeros_like(z)


def self_check() -> None:
    ctl = UBFLoadController()
    x = torch.tensor([0.2, 0.1])
    xd = torch.zeros(2)
    v = torch.zeros(2)
    vd = torch.zeros(2)
    F, eta, z = ctl(x, xd, v, vd, m_L=5.0)
    assert F.shape == (2,)
    assert not bool(torch.isnan(F).any())
    assert float(eta) >= 0.0
    print("ubf_controller: OK")


if __name__ == "__main__":
    self_check()
