"""cuda:0 + AMP FP16 + optional DDP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DeviceContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    amp: bool = True
    is_main: bool = True

    @property
    def use_ddp(self) -> bool:
        return self.world_size > 1


def init_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def get_device(prefer: str = "cuda:0") -> torch.device:
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_device_context(
    prefer: str = "cuda:0",
    amp: bool | None = None,
) -> DeviceContext:
    rank, world_size, local_rank = init_distributed()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda:0")
        if world_size == 1:
            torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    use_amp = bool(amp if amp is not None else device.type == "cuda")
    return DeviceContext(
        device=device,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank if world_size > 1 else 0,
        amp=use_amp and device.type == "cuda",
        is_main=(rank == 0),
    )


def to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def wrap_ddp(model: torch.nn.Module, ctx: DeviceContext) -> torch.nn.Module:
    model = model.to(ctx.device)
    if ctx.use_ddp:
        model = DDP(
            model,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=False,
        )
    return model


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def print_banner(
    ctx: DeviceContext,
    *,
    projection: str = "anticipatory",
    projection_detach: bool = True,
    epsilon: float | None = None,
    delta: float | None = None,
    violation: float | None = None,
) -> None:
    if not ctx.is_main:
        return
    gpu = "n/a"
    if ctx.device.type == "cuda" and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(ctx.device)
    lines = [
        "",
        "SECDO Training",
        "",
        "GPU:",
        f"{ctx.device} {gpu}",
        "",
        "Projection:",
        projection,
        "",
        "gradient through projection:",
        "disabled" if projection_detach else "ENABLED (BAD)",
        "",
        "AMP:",
        "FP16" if ctx.amp else "FP32",
        "",
    ]
    if epsilon is not None:
        lines += ["epsilon:", f"{epsilon:.6f}", ""]
    if delta is not None:
        lines += ["delta:", f"{delta:.6f}", ""]
    if violation is not None:
        lines += ["violation:", f"{violation:.6f}", ""]
    print("\n".join(lines), flush=True)


print_training_banner = print_banner


def maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
