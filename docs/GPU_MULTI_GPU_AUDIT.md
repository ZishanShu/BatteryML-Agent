# GPU inventory and multi-GPU audit

Milestone E. Two questions, answered separately:

1. **Does the platform understand the machine's GPUs?** Yes — see
   *Inventory* below. Enumeration is real, the count is discovered, and any
   single GPU can be selected.
2. **Can a BatteryML run use several GPUs at once?** **No, and this
   repository must not pretend otherwise.** The reasons are structural and
   are set out in *Audit* below, with the work that a real implementation
   would require.

Every statement here is tagged per `CLAUDE.md` §0. Nothing in this document
was established by running the pipeline: this environment has no torch, no
GPU and no data.

---

## Inventory — implemented

`batteryml_web/core/device_inventory.py` enumerates CUDA devices by running

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits
```

as an argv list with `shell=False`, and parsing the rows positionally.

* **The GPU count is discovered, never assumed.** `[STATIC]` The parser
  returns one row per line of output; `device_choices()` builds `cpu` plus
  one entry per enumerated device. An eight-GPU server offers eight; a
  two-GPU machine offers two; a CPU-only machine offers `cpu` alone. There is
  no `8` anywhere in the code (`scripts/verify/check_runtime_contract.py`,
  DEV-01, asserts the count follows the output for 0, 1, 3 and 8 rows).
* **Free memory is reported as observed.** `[STATIC]` A column the driver
  reports as `[N/A]` becomes `None` and is displayed as `unknown`, never as
  `0`. The Agent Workflow page shows the per-device table and warns with the
  selected GPU's free memory before a run.
* **Enumeration failure is a state, not a crash.** `[STATIC]` A missing
  `nvidia-smi`, a timeout, a non-zero exit or an unparsable row all return an
  inventory with `error` set; the page shows that text and still offers a
  manual `cuda:0` so an explicit GPU choice stays possible.
* **A malformed row raises rather than being dropped.** `[STATIC]` Silently
  skipping a row would under-report the machine's GPUs, which is worse than
  failing with the offending line quoted.

Torch is deliberately *not* imported to enumerate devices: creating a CUDA
context on a shared server costs memory on a machine that is already short of
it, and `nvidia-smi` answers the question without allocating anything.

---

## Audit — why there is no multi-GPU run

### What the code actually does with `--device`

`[STATIC]` The device travels as a single string, end to end:

| Step | Code | What happens |
| --- | --- | --- |
| CLI | `bin/batteryml.py` `run_parser.add_argument("--device", default="cpu")` | one string, no list form, no `--nproc`, no launcher flag |
| Pipeline | `batteryml/pipeline.py:137` `model = model.to(device)` | one `torch.device` |
| Pipeline | `batteryml/pipeline.py:51,97` `build_dataset(self.config, device)` → `dataset.to(device)` | one `torch.device` |
| Data | `batteryml/data/databundle.py:67-74` `DataBundle.to()` | moves `train_data`, `test_data` **and** the fitted transformations onto that one device |
| Model | `batteryml/models/nn_model.py:104` `def to(self, device)` → `nn.Module.to` | one `torch.device` |

`"cuda:0,1"` is not a torch device. Passing it would fail inside
`.to(device)` partway through a run, after preprocessing and dataset
construction have already been paid for. `build_execution_plan` now refuses
it up front (`batteryml_web/core/device_inventory.validate_device`).

### What is absent

`[STATIC]` A repository-wide search for `DataParallel`,
`DistributedDataParallel`, `torch.distributed`, `init_process_group`,
`torchrun`, `nproc_per_node`, `device_ids` and `LOCAL_RANK` across
`batteryml/`, `batteryml_web/`, `bin/`, `configs/` and `scripts/` returns
**nothing**. There is no partial multi-GPU support to switch on: it would
have to be written.

### Why it would not solve the problem it appears to solve

The development server has 8 × RTX 3090 with roughly 2 GB free on each.
The intuition "8 × 2 GB = 16 GB" is wrong for both mechanisms:

* **DataParallel** replicates the *whole model* onto every GPU each forward
  pass and scatters only the input batch. Every replica needs its own full
  copy of the parameters, gradients and optimizer state. It lowers per-GPU
  *activation* memory, never per-GPU *model* memory.
* **DistributedDataParallel** is the same story with one process per GPU:
  each process holds a complete model and a complete optimizer state.

`[STATIC]` And in this codebase the binding constraint is neither: it is
`DataBundle.to(device)`, which places the **entire train and test feature
tensors** on a single device before training starts. Sharding the batch does
not shard that residency. A dataset that does not fit in 2 GB does not fit
under DP or DDP either. Only tensor/parameter sharding (FSDP, ZeRO-3) plus a
streaming dataset would change that answer, and both are far larger changes
than this milestone.

### Model-specific constraints

* **BatLiNet** `[STATIC]` (`batteryml_web/models/batlinet.py`). Its `fit()`
  takes the device *from the data* (`dataset.train_data.feature.device`), and
  `_training_support()` builds each example's inter-cell feature by indexing
  the **full** training feature tensor — `target.unsqueeze(1) - refs`, with
  `refs` sampled from all training cells. Under DataParallel the batch is
  scattered but that full tensor lives on one device: the replicas would
  either fault on a cross-device index or, if the tensor were replicated,
  each replica would draw its support set from a different sampling stream.
  The second case is worse than a crash — it would train a *different model*
  under the same config and seed, silently. That is a change to the
  scientific semantics this branch is forbidden to make.
* **CPTransformer** `[STATIC]` (`batteryml_web/models/cptransformer.py`)
  subclasses `NNModel`, whose `fit()` iterates a plain `DataLoader` over data
  already resident on the device. It is the more DDP-friendly of the two, but
  DDP still multiplies the effective batch size by the world size unless
  `train_batch_size` is divided — and `train_batch_size` is a declared value
  in the formal config, so changing it changes the benchmark.
* **`torch.compile`** `[STATIC]` (`batteryml/pipeline.py:134-135`) wraps the
  model on torch ≥ 2 before `.to(device)`. Any DP/DDP wrapper has to be
  ordered against it deliberately; getting that wrong is a silent
  recompilation-per-step performance trap rather than an error.

### Determinism

`[STATIC]` `CLAUDE.md` §8 requires that identical config + seed + data
fingerprint produce identical `metrics.csv`. `set_seed` seeds one process.
DDP would introduce per-rank seeding, a distributed sampler and gradient
all-reduce ordering, each of which changes the numbers. Reproducing the
single-GPU result would then be a separate piece of work, not a side effect.

---

## Verdict

**Multi-GPU is not implemented, and no multi-GPU Run option is exposed.**
`[STATIC]` What ships is:

* real GPU enumeration and single-GPU selection (above);
* an explicit refusal of multi-device strings, with the reason, at the
  planner boundary rather than deep inside a run.

This is a deliberate stop, not an omission: implementing DP or DDP here
would change batch semantics, change BatLiNet's support-set construction,
change determinism, and require edits to upstream `batteryml/` core — all of
which are out of scope for this branch and none of which would give a run
more memory than one GPU has.

## What a real implementation would require

Roughly in order; each item is a separate, reviewable change.

1. **Decide the goal.** "Use the idle GPUs" (throughput) and "train a model
   that does not fit on one GPU" (capacity) are different problems with
   different solutions. Only the second needs sharding; the first is better
   served by running independent seeds/configs on separate GPUs.
2. **Prefer multi-run parallelism first.** `[ASSUMED]` N independent
   single-GPU runs (one per seed or per config) use N GPUs with no change to
   any model, no change to batch semantics and no change to determinism.
   This is very likely the right answer for a benchmark sweep, and would be
   verified by comparing one such run's `metrics.csv` against the current
   single-GPU baseline.
3. **Stream the dataset** before anything else, if capacity is the goal.
   `DataBundle.to(device)` must stop being whole-dataset residency, or no
   parallelism strategy helps. This is an upstream `batteryml/` change and
   needs approval per `CLAUDE.md` §4, recorded in `docs/UPSTREAM_PATCHES.md`.
4. **DDP for CPTransformer only**, behind an explicit structured execution
   option: a `torchrun` launcher in the CLI wrapper, a `DistributedSampler`,
   rank-aware seeding, rank-0-only checkpointing and artifact writing, and
   `train_batch_size` interpreted per-rank so the *effective* batch matches
   the config.
5. **Prove equivalence.** A DDP run and a single-GPU run of the same config
   and seed must produce the same `metrics.csv`, or the difference must be
   recorded in `run_metadata.json` as a declared non-determinism
   (`CLAUDE.md` §8). Without this the benchmark numbers are not comparable.
6. **Leave BatLiNet single-GPU** until its inter-cell support sampling has a
   defined distributed semantics. Its configured `train_support_size` /
   `test_support_size` are scientific settings; a distributed variant is a
   modelling decision for the authors, not a deployment detail.

Until at least steps 3–5 exist and are verified on the server, the honest
interface is the one in this branch: one GPU, chosen explicitly, with its
free memory shown.
