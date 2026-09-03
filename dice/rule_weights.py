"""Rule weight utilities."""

from __future__ import annotations

import torch

from dice.local_rules import LocalRules


def freeze_weights(rules: LocalRules) -> None:
    for p in rules.parameters():
        p.requires_grad_(False)


def softplus_normalize(rules: LocalRules) -> dict[str, float]:
    ws = {
        "sep": float(torch.nn.functional.softplus(rules.w_separation)),
        "ali": float(torch.nn.functional.softplus(rules.w_alignment)),
        "coh": float(torch.nn.functional.softplus(rules.w_cohesion)),
        "goal": float(torch.nn.functional.softplus(rules.w_goal)),
    }
    s = sum(ws.values()) + 1e-8
    return {k: v / s for k, v in ws.items()}
