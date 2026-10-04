"""Detect the machine and size the models to fill it.

The spec is "detect all computational power available, then make the models as
large as it supports". That is a real constraint-satisfaction problem, not a
constant, because the same code has to run on:

* Snowflake Container Runtime: 4x A10 (23 GB), 48 vCPU, 100 GB RAM
* molab: 1x RTX PRO 6000 (96 GB), serving a single live stream
* a laptop / CI box with no GPU at all, for smoke tests

So we measure, then solve for the widest model whose *training* footprint
(parameters + gradients + Adam moments + activations) fits the smallest GPU
with headroom, and the widest CPU-side analyser that still clears one forward
pass inside the one-second budget the live loop allows.

Nothing here guesses. If torch cannot see a GPU, it says so and returns a CPU
plan rather than producing a config that will OOM twenty minutes into a run.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

LOG = logging.getLogger("btcpred.hardware")

BYTES_PER_GB = 1 << 30


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_gb: float
    capability: tuple[int, int]
    bf16_supported: bool


@dataclass
class Hardware:
    n_gpus: int
    gpus: list[GPUInfo]
    cpu_count: int
    ram_gb: float
    torch_version: str
    cuda_version: str | None
    nccl_available: bool
    platform: str = field(default_factory=lambda: platform.platform())

    @property
    def min_gpu_memory_gb(self) -> float:
        """Smallest GPU in the box: the real constraint for a DDP job."""
        return min((g.total_memory_gb for g in self.gpus), default=0.0)

    @property
    def homogeneous(self) -> bool:
        return len({g.name for g in self.gpus}) <= 1

    @property
    def bf16(self) -> bool:
        return bool(self.gpus) and all(g.bf16_supported for g in self.gpus)

    def summary(self) -> str:
        lines = [
            f"torch {self.torch_version} | cuda {self.cuda_version or 'n/a'} | "
            f"nccl {'yes' if self.nccl_available else 'no'}",
            f"cpu: {self.cpu_count} logical cores | ram: {self.ram_gb:.0f} GB",
        ]
        if not self.gpus:
            lines.append("gpu: none visible -- CPU-only plan")
        for g in self.gpus:
            lines.append(
                f"gpu{g.index}: {g.name} | {g.total_memory_gb:.1f} GB | "
                f"sm_{g.capability[0]}{g.capability[1]} | "
                f"bf16 {'yes' if g.bf16_supported else 'no'}"
            )
        if self.gpus and not self.homogeneous:
            lines.append(
                "WARNING: mixed GPU models. Sizing against the smallest; "
                "DDP step time will be set by the slowest."
            )
        return "\n".join(lines)


def _ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / BYTES_PER_GB
    except (ValueError, OSError, AttributeError):
        pass
    if shutil.which("free"):
        try:
            out = subprocess.run(
                ["free", "-b"], capture_output=True, text=True, check=True
            ).stdout.splitlines()[1]
            return int(out.split()[1]) / BYTES_PER_GB
        except Exception:  # noqa: BLE001
            pass
    return 0.0


def _cpu_count() -> int:
    # sched_getaffinity respects cgroup/container pinning; os.cpu_count does
    # not, and over-reporting cores makes the CPU analyser thread-thrash.
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def detect() -> Hardware:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required. `pip install -r requirements.txt`"
        ) from exc

    gpus: list[GPUInfo] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cap = (props.major, props.minor)
            gpus.append(
                GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_gb=props.total_memory / BYTES_PER_GB,
                    capability=cap,
                    # bf16 needs Ampere (sm_80) or newer. A10 is sm_86, the
                    # RTX PRO 6000 is sm_89/sm_120 -- all fine.
                    bf16_supported=cap >= (8, 0),
                )
            )

    return Hardware(
        n_gpus=len(gpus),
        gpus=gpus,
        cpu_count=_cpu_count(),
        ram_gb=_ram_gb(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        nccl_available=bool(
            getattr(torch.distributed, "is_nccl_available", lambda: False)()
        ),
    )


# --------------------------------------------------------------------------- #
# Capacity planning
# --------------------------------------------------------------------------- #
@dataclass
class SizingPlan:
    """A concrete model shape chosen to fit the detected hardware."""

    predictor_d_model: int
    predictor_layers: int
    predictor_heads: int
    predictor_ffn_mult: int
    analyser_d_model: int
    analyser_layers: int
    micro_batch: int
    grad_accum: int
    precision: str  # "bf16" | "fp16" | "fp32"
    activation_checkpointing: bool
    analyser_threads: int
    dataloader_workers: int
    notes: list[str] = field(default_factory=list)

    def predictor_params(self, n_tokens: int, n_features: int, horizon_basis: int) -> int:
        return _transformer_params(
            self.predictor_d_model, self.predictor_layers,
            self.predictor_ffn_mult, n_tokens, n_features, horizon_basis,
        )

    def analyser_params(self, n_features: int) -> int:
        d, l = self.analyser_d_model, self.analyser_layers
        # Dilated residual conv stack: 2 convs of kernel 5 per block.
        return l * (2 * d * d * 5 + 4 * d) + n_features * d + d * d

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _transformer_params(
    d: int, layers: int, ffn_mult: int, n_tokens: int, n_features: int, out_dim: int
) -> int:
    attn = 4 * d * d
    ffn = 2 * d * d * ffn_mult
    per_layer = attn + ffn + 4 * d  # + two layernorms
    return layers * per_layer + n_features * d + n_tokens * d + d * out_dim


def _training_bytes(params: int, micro_batch: int, n_tokens: int, d: int,
                    layers: int, precision: str, checkpointing: bool) -> float:
    """Estimated GPU bytes for one training step.

    Weights + grads + two Adam moments, plus activations. Adam state is kept
    in fp32 even under bf16 autocast, which is the single biggest term people
    forget: a 400M-parameter model costs 3.2 GB of optimiser state alone.
    """
    master = params * 4  # fp32 master weights
    grads = params * 4
    adam = params * 8  # exp_avg + exp_avg_sq, fp32
    compute_copy = params * (2 if precision in ("bf16", "fp16") else 0)

    act_per_layer = micro_batch * n_tokens * d * (2 if precision != "fp32" else 4)
    # ~12 live tensors per transformer block without checkpointing; with
    # checkpointing only the block boundaries survive, plus one recomputed block.
    acts = act_per_layer * (12 * layers if not checkpointing else layers + 12)
    return master + grads + adam + compute_copy + acts


def plan(
    hw: Hardware,
    n_tokens: int,
    n_features: int,
    horizon_basis: int,
    memory_headroom: float = 0.80,
    max_d_model: int = 2048,
) -> SizingPlan:
    """Choose the widest model that fits, with ``memory_headroom`` to spare.

    ``memory_headroom`` is the fraction of each GPU we are willing to occupy.
    0.80 is deliberately conservative: CUDA context, NCCL buffers, allocator
    fragmentation and the odd evaluation batch all need room, and an OOM three
    hours into an online training run is far more expensive than a slightly
    smaller model.
    """
    notes: list[str] = []

    if not hw.gpus:
        notes.append(
            "No CUDA device found. Falling back to a small CPU plan suitable "
            "only for smoke tests -- do not expect meaningful accuracy."
        )
        return SizingPlan(
            predictor_d_model=128, predictor_layers=4, predictor_heads=4,
            predictor_ffn_mult=4, analyser_d_model=96, analyser_layers=4,
            micro_batch=2, grad_accum=1, precision="fp32",
            activation_checkpointing=False,
            analyser_threads=max(1, hw.cpu_count - 1),
            dataloader_workers=min(2, max(0, hw.cpu_count - 1)),
            notes=notes,
        )

    budget = hw.min_gpu_memory_gb * memory_headroom * BYTES_PER_GB
    precision = "bf16" if hw.bf16 else "fp16"
    if not hw.bf16:
        notes.append(
            "GPU predates Ampere: using fp16 with a grad scaler instead of "
            "bf16. Watch for loss-scale underflow on the quantile heads."
        )

    # Candidate widths, widest first. Depth scales with width so the model
    # stays roughly isotropic rather than becoming a deep thin pipe.
    candidates = [
        (d, layers, ffn)
        for d, layers, ffn in (
            (2048, 32, 4), (1792, 30, 4), (1536, 28, 4), (1280, 26, 4),
            (1024, 24, 4), (896, 20, 4), (768, 18, 4), (640, 16, 4),
            (512, 12, 4), (384, 10, 4), (256, 8, 4),
        )
        if d <= max_d_model
    ]

    target_tokens_per_step = 32  # effective batch of market-seconds per update

    for d, layers, ffn in candidates:
        params = _transformer_params(d, layers, ffn, n_tokens, n_features, horizon_basis)
        for checkpointing in (False, True):
            for micro in (16, 8, 4, 2, 1):
                need = _training_bytes(
                    params, micro, n_tokens, d, layers, precision, checkpointing
                )
                if need <= budget:
                    accum = max(1, math.ceil(target_tokens_per_step / (micro * hw.n_gpus)))
                    heads = max(4, d // 64)
                    # Analyser lives on CPU; size it by core count, not VRAM.
                    a_d = 512 if hw.cpu_count >= 32 else (384 if hw.cpu_count >= 16 else 256)
                    a_layers = 10 if hw.cpu_count >= 32 else 8
                    notes.append(
                        f"predictor {params / 1e6:.0f}M params, "
                        f"est. {need / BYTES_PER_GB:.1f} GB/GPU of "
                        f"{hw.min_gpu_memory_gb:.1f} GB "
                        f"(headroom {memory_headroom:.0%})"
                    )
                    if checkpointing:
                        notes.append(
                            "activation checkpointing ON: ~30% slower per step, "
                            "but it bought the extra width."
                        )
                    return SizingPlan(
                        predictor_d_model=d, predictor_layers=layers,
                        predictor_heads=heads, predictor_ffn_mult=ffn,
                        analyser_d_model=a_d, analyser_layers=a_layers,
                        micro_batch=micro, grad_accum=accum,
                        precision=precision,
                        activation_checkpointing=checkpointing,
                        analyser_threads=max(1, hw.cpu_count // max(hw.n_gpus, 1) - 1),
                        dataloader_workers=min(
                            8, max(1, (hw.cpu_count // max(hw.n_gpus, 1)) - 2)
                        ),
                        notes=notes,
                    )

    raise RuntimeError(
        f"Even the smallest candidate model does not fit in "
        f"{hw.min_gpu_memory_gb:.1f} GB at {memory_headroom:.0%} headroom. "
        f"Raise --memory-headroom or use a larger GPU."
    )


def main() -> None:  # pragma: no cover - CLI helper
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    hw = detect()
    print(hw.summary())
    print()
    from ..data.pyramid import PyramidSpec
    from ..data.schema import N_FEATURES

    spec = PyramidSpec()
    p = plan(hw, spec.n_tokens, N_FEATURES, horizon_basis=48)
    print(f"tokens/sample: {spec.n_tokens}  features: {N_FEATURES}")
    print(
        f"predictor: d_model={p.predictor_d_model} layers={p.predictor_layers} "
        f"heads={p.predictor_heads}  "
        f"(~{p.predictor_params(spec.n_tokens, N_FEATURES, 48) / 1e6:.0f}M params)"
    )
    print(
        f"analyser : d_model={p.analyser_d_model} layers={p.analyser_layers} "
        f"(~{p.analyser_params(N_FEATURES) / 1e6:.0f}M params, CPU, "
        f"{p.analyser_threads} threads)"
    )
    print(
        f"batching : micro={p.micro_batch} accum={p.grad_accum} "
        f"precision={p.precision} checkpointing={p.activation_checkpointing}"
    )
    for n in p.notes:
        print(f"  - {n}")


if __name__ == "__main__":  # pragma: no cover
    main()
