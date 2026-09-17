"""The process torchrun launches on each rank.

One of these runs per GPU. Its job is to set up the distributed context
correctly, train, and make sure that everything a run produces exactly once
is produced exactly once.

The rules it exists to enforce:

* **rank-local devices.** ``CUDA_VISIBLE_DEVICES`` has already renumbered the
  selected GPUs, so rank *i* uses ``cuda:i``. No physical index appears here.
* **sharded data, not replicated.** A ``DistributedSampler`` gives each rank
  its own slice, and ``set_epoch`` is called every epoch so the shuffle
  differs between epochs and agrees across ranks. Without that call every
  epoch sees the same order — a silent change to training.
* **rank 0 writes.** Manifest transitions, the predictions artifact, metrics,
  figures and the canonical checkpoint are written by rank 0 alone. Four
  ranks writing one path is how an artifact becomes whichever rank finished
  last.
* **evaluation happens once.** After training, ranks synchronise and rank 0
  evaluates the finished model on the whole test set — the same procedure a
  single-GPU run uses, so the metric means the same thing. Sharded evaluation
  plus all-gather would give the same numbers only if every rank saw a
  disjoint, complete partition, and getting that subtly wrong changes a
  reported metric. The simplest correct option is chosen deliberately.
* **checkpoints are portable.** DDP wraps the model, so its ``state_dict``
  keys gain a ``module.`` prefix. What is saved is the *unwrapped* model's
  state, so a distributed checkpoint loads into a single-GPU evaluation and
  into a resume without a key-rewriting step.

No Streamlit. Import-safe: nothing here runs on import, so the module can be
inspected by tests without a process group.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

#: Written by rank 0 so a reader can tell a distributed run at a glance.
DISTRIBUTED_MARKER = "distributed.json"


def rank_info(environ: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """Rank, local rank and world size, from torchrun's environment.

    Pure and injectable so the mapping can be tested without launching
    anything. Defaults describe a single-process run, which is what a
    directly-invoked runner is.
    """
    env = environ if environ is not None else os.environ
    def _int(name: str, default: int) -> int:
        try:
            return int(env.get(name, default))
        except (TypeError, ValueError):
            return default

    return {
        "rank": _int("RANK", 0),
        "local_rank": _int("LOCAL_RANK", 0),
        "world_size": _int("WORLD_SIZE", 1),
    }


def is_primary(info: Optional[Dict[str, int]] = None) -> bool:
    """Whether this process performs the shared, write-once side effects."""
    return (info or rank_info())["rank"] == 0


def select_device(info: Optional[Dict[str, int]] = None) -> str:
    """The device this rank trains on.

    ``cuda:{local_rank}`` after ``CUDA_VISIBLE_DEVICES`` renumbering, or cpu
    when no CUDA is visible — which is what makes a gloo run on a laptop use
    the same code path as an NCCL run on the server.
    """
    import torch

    resolved = info or rank_info()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return f"cuda:{resolved['local_rank']}"
    return "cpu"


def default_backend() -> str:
    """NCCL where there are GPUs, gloo otherwise."""
    import torch

    return "nccl" if torch.cuda.is_available() else "gloo"


def init_process_group(backend: str = "", timeout_s: int = 1800) -> Dict[str, Any]:
    """Join the process group and pin this rank's device."""
    import datetime

    import torch
    import torch.distributed as dist

    info = rank_info()
    chosen = backend or default_backend()
    if not dist.is_initialized():
        dist.init_process_group(
            backend=chosen,
            timeout=datetime.timedelta(seconds=timeout_s),
        )
    device = select_device(info)
    if device.startswith("cuda"):
        torch.cuda.set_device(info["local_rank"])
    return {**info, "backend": chosen, "device": device}


def destroy_process_group() -> None:
    """Leave the group. Safe to call when no group was ever created."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_sampler(dataset: Any, info: Optional[Dict[str, int]] = None, *, seed: int = 0):
    """A DistributedSampler giving this rank its own shard.

    ``drop_last=False`` keeps every sample in an epoch; the sampler pads the
    last shard so all ranks step the same number of times, which is what DDP's
    gradient synchronisation requires.
    """
    from torch.utils.data.distributed import DistributedSampler

    resolved = info or rank_info()
    return DistributedSampler(
        dataset,
        num_replicas=resolved["world_size"],
        rank=resolved["rank"],
        shuffle=True,
        seed=int(seed),
        drop_last=False,
    )


def wrap_model(model: Any, info: Optional[Dict[str, int]] = None) -> Any:
    """Wrap in DistributedDataParallel, on this rank's device."""
    import torch
    from torch.nn.parallel import DistributedDataParallel

    resolved = info or rank_info()
    device = select_device(resolved)
    model = model.to(device)
    if device.startswith("cuda"):
        return DistributedDataParallel(
            model, device_ids=[resolved["local_rank"]],
            output_device=resolved["local_rank"],
        )
    # gloo/CPU: no device_ids, which is what makes CPU tests of this path
    # possible without pretending a GPU exists.
    return DistributedDataParallel(model)


def unwrap(model: Any) -> Any:
    """The underlying module, whether or not ``model`` is DDP-wrapped.

    Used before saving so a checkpoint never carries ``module.`` prefixes —
    an interoperability trap that would make a distributed checkpoint fail to
    load into single-GPU evaluation or into a resume.
    """
    return getattr(model, "module", model)


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def rank_log_name(info: Optional[Dict[str, int]] = None) -> str:
    """Per-rank log filename, so ranks never interleave into one file."""
    return f"rank_{(info or rank_info())['rank']}.log"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """The runner's own arguments, mirroring the launch contract."""
    parser = argparse.ArgumentParser(
        prog="batteryml-distributed-runner",
        description="Rank-local training process launched by torchrun.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--per-rank-batch-size", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--backend", default="")
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help=(
            "Join the group, report rank/device/world size as JSON on stdout, "
            "then leave. Used to verify a distributed launch without training."
        ),
    )
    return parser.parse_args(argv)


def describe(args: argparse.Namespace) -> Dict[str, Any]:
    """What this rank is, once the group is up.

    The payload the ``--describe-only`` mode prints. It is how a launch can be
    proved — world size, per-rank device and the rank mapping — without
    running a training job.
    """
    context = init_process_group(args.backend)
    try:
        barrier()
        return {
            "rank": context["rank"],
            "local_rank": context["local_rank"],
            "world_size": context["world_size"],
            "backend": context["backend"],
            "device": context["device"],
            "is_primary": context["rank"] == 0,
            "per_rank_batch_size": args.per_rank_batch_size,
            "global_batch_size": args.global_batch_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
    finally:
        destroy_process_group()


def main(argv: Optional[List[str]] = None) -> int:
    import json

    args = parse_args(argv)
    if args.describe_only:
        print(json.dumps(describe(args)), flush=True)
        return 0

    # Training through this runner is not wired to the BatteryML pipeline yet:
    # doing so needs a model whose DDP-safety has been audited against its own
    # source, which cannot be done in an environment where that source and the
    # GPUs are both absent. Refusing loudly is the honest behaviour; a runner
    # that silently trained something would be worse than one that says this.
    print(
        "batteryml-distributed-runner: training is not wired to the pipeline "
        "in this build. The launch contract, rank mapping, sharding and "
        "rank-0 rules are implemented and tested; see docs/DDP_CONTRACT.md "
        "for what remains before a model may be marked DDP-capable.",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
