"""Process-group setup for the two-device training topology.

This job is unusual: the Market Analyser lives on CPU and the Price Predictors
live on GPU, and they need *different* collectives.

* The analyser is shared — one logical model — so its gradients must be
  all-reduced across every rank. CPU tensors need the **gloo** backend; NCCL
  cannot reduce them.
* In ``ddp`` mode the predictor is also shared and all-reduces over **nccl**.
  In ``ensemble`` mode each rank trains a *different* predictor, so the
  predictor must emphatically *not* be all-reduced — only the analyser is.

PyTorch supports exactly this via a ``cpu:gloo,cuda:nccl`` backend string: one
process group, two transports, each collective routed by the device of its
tensors. That is what :func:`init` sets up when CUDA is present.

Launch with ``torchrun``; everything here reads the standard environment
variables it sets and degrades to a sane single-process path when they are
absent, so the same script runs under ``python -m`` for smoke tests.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

LOG = logging.getLogger("btcpred.dist")


@dataclass(frozen=True)
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init(timeout_minutes: int = 30) -> DistInfo:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    distributed = world_size > 1
    if distributed:
        backend = "cpu:gloo,cuda:nccl" if use_cuda else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=dt.timedelta(minutes=timeout_minutes),
        )
        LOG.info(
            "rank %d/%d initialised on %s (backend=%s)",
            rank, world_size, device, backend,
        )
    return DistInfo(rank, local_rank, world_size, device, distributed)


def barrier(info: DistInfo) -> None:
    if info.distributed:
        dist.barrier()


def shutdown(info: DistInfo) -> None:
    if info.distributed and dist.is_initialized():
        dist.destroy_process_group()


def all_gather_object(info: DistInfo, obj):
    """Gather a picklable object from every rank onto every rank."""
    if not info.distributed:
        return [obj]
    out = [None] * info.world_size
    dist.all_gather_object(out, obj)
    return out


def all_reduce_mean(info: DistInfo, value: float) -> float:
    if not info.distributed:
        return value
    t = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / info.world_size)
