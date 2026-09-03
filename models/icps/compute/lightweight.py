"""FP16 / eval-path lightweighting helper for AC-DSGF-sized MLPs."""

from __future__ import annotations

import torch
import torch.nn as nn


def to_fp16_module(module: nn.Module) -> nn.Module:
    return module.half()


def quantize_dynamic_linear(module: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization for Linear layers (CPU)."""
    return torch.quantization.quantize_dynamic(
        module, {nn.Linear}, dtype=torch.qint8
    )


@torch.no_grad()
def accuracy_proxy(fp_out: torch.Tensor, other_out: torch.Tensor) -> float:
    """Relative L2 agreement as stand-in for 'precision loss'."""
    num = (fp_out.float() - other_out.float()).norm()
    den = fp_out.float().norm().clamp(min=1e-8)
    return float(1.0 - (num / den).clamp(0, 1))


def self_check() -> None:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    x = torch.randn(16, 32)
    y = net(x)
    net_h = to_fp16_module(nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8)))
    net_h.load_state_dict({k: v.half() for k, v in net.state_dict().items()})
    y_h = net_h(x.half()).float()
    agree = accuracy_proxy(y, y_h)
    assert agree >= 0.97, f"FP16 agree {agree:.3f} < 0.97 (~3% loss budget)"
    print(f"lightweight: OK (fp16_agree={agree:.4f})")


if __name__ == "__main__":
    self_check()
