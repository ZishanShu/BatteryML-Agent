# Distributed (DDP) contract — what is built, and what is still gated

Milestone B asked for true multi-GPU execution. This records exactly what
exists, what it guarantees, and what must happen on the server before any
model may be marked DDP-capable.

**Status: the launch contract and the distributed runtime rules are
implemented and tested. No model is marked `supports_multi_gpu = true`, and
true multi-GPU training is therefore still NOT available to a user.** The
gate is deliberate, not an oversight; §6 says what lifts it.

---

## 1. Launch contract (`core/distributed_plan.py`)

A structured selection becomes a real distributed launch:

```
selection: {"mode": "multi_gpu", "gpu_ids": [5, 6]}

env:  CUDA_VISIBLE_DEVICES=5,6
      OMP_NUM_THREADS=1
argv: torchrun --standalone --nnodes=1 --nproc_per_node=2
        -m batteryml_web.core.distributed_runner
        --config <config> --workspace <run dir> --seed <n>
        --per-rank-batch-size <b> --global-batch-size <B>
```

`cuda:0,1` is never produced. It is not a torch device and never was;
emitting it is how a multi-GPU request became a crash inside a run that had
already started. A test asserts no argv element contains a comma-separated
device, and a refused plan carries an **empty** argv so there is nothing to
execute by accident.

The physical GPU ids appear only in `CUDA_VISIBLE_DEVICES`. That renumbers
devices for the child, so rank *i* uses `cuda:{i}` — which is why selecting
GPUs 5 and 6 behaves exactly like selecting 0 and 1, and why no physical
index is embedded in the command.

argv list, explicit environment dict, `shell=False` — the same rules as every
other execution path here.

## 2. Batch-size semantics

**`train_batch_size` in a config is the GLOBAL batch size.** It is divided by
the number of ranks, so a 2-GPU run takes the same number of samples per
optimizer step as a 1-GPU run and the optimization is unchanged.

The rejected alternative is per-rank semantics, where `world_size` GPUs
silently multiply the effective batch by `world_size`. That changes what Adam
averages over, and therefore the result, without anything in the UI saying
so.

An indivisible batch is an **error**, not a rounding exercise: `33` across 2
ranks is refused with a message naming both fixes (change the GPU count, or
change the config). Rounding would quietly change the effective batch.

## 3. Distributed data

`DataBundle.to(device)` places the *entire* dataset on one device
(`batteryml/data/databundle.py:67-74`). Naive DDP that replicated the bundle
per rank would keep that memory cost on every GPU while adding
synchronisation — the opposite of the reason multi-GPU was wanted.

The runner therefore uses a `DistributedSampler` over CPU-resident data, one
shard per rank, with `set_epoch(epoch)` called every epoch so the shuffle
differs between epochs and agrees across ranks. Without that call, every
epoch replays one order — a silent change to training.

`drop_last=False`: the sampler pads the final shard so every rank takes the
same number of steps, which DDP's gradient synchronisation requires.

## 4. Rank rules

| Concern | Rule |
| --- | --- |
| Device | `cuda:{local_rank}` after renumbering; `cpu` when no CUDA is visible |
| Backend | NCCL where GPUs exist, gloo otherwise — the same code path either way |
| Manifest transitions | rank 0 only |
| Predictions artifact, metrics, figures, bundle | rank 0 only |
| Canonical checkpoint | rank 0 only |
| Per-rank logs | `rank_0.log`, `rank_1.log`, … so ranks never interleave |

Exactly one rank is primary; a test asserts that over a simulated world of 4.

## 5. Evaluation and checkpoints

**Evaluation runs once.** After training the ranks synchronise and rank 0
evaluates the finished model over the whole test set — the identical
procedure a single-GPU run uses, so the metric means the same thing. Sharded
evaluation with all-gather would agree only if the partition were exactly
disjoint and complete, and getting that subtly wrong silently changes a
reported metric. The simplest correct option was chosen on purpose.

**Checkpoints are saved unwrapped.** DDP's `state_dict` keys carry a
`module.` prefix; storing that would make a distributed checkpoint fail to
load into single-GPU evaluation or into a resume. `unwrap()` is applied
before saving, and a test asserts no stored key begins with `module.`. The
resume schema from Milestone A is used unchanged, so a DDP checkpoint is
resumable by the same code path.

## 6. Capability gating — why no model is enabled

| Model | DDP-capable? | Reason |
| --- | --- | --- |
| CPTransformer | **no — gated** | its network source lives in `third_party/BatteryLife`, absent from the environment this was audited in, so its forward and loss could not be read and DDP-safety could not be established |
| BatLiNet | **no — scientific** | it draws an inter-cell support set per batch from the full training tensor (`batlinet.py:884`). Sharding changes which comparisons the model sees, and so the objective; replicating the tensor removes the memory benefit and still changes the sampling distribution. This needs a redesign of the support semantics, not a sampler |
| CNN / LSTM / MLP / Transformer | **no — unverified** | these share the auditable `NNModel.fit` and are the most likely first candidates, but none has been run under DDP on real GPUs |
| sklearn / XGBoost | **no — n/a** | not torch models |

These reasons are stored per model in `run_policy.MULTI_GPU_BLOCKERS` and
surfaced by `multi_gpu_capability`. The UI reads that; it names no model.

### What would lift the gate for CPTransformer

1. read the network's source on the server and confirm nothing in its forward
   or loss assumes a global batch, a fixed rank count, or cross-sample state;
2. wire the runner's training path to the BatteryML pipeline for that model;
3. run the two-GPU smoke check in §7 and confirm world size, ranks, one
   predictions artifact and one metric set;
4. compare a 1-GPU and a 2-GPU run per §8 before enabling it in
   `MULTI_GPU_CAPABLE_MODELS`.

Until step 4, `supports_multi_gpu` stays false. **Marking it true earlier
would make an unverified model appear supported, which is the failure this
milestone was written to avoid.**

## 7. What was actually verified here

`[VERIFIED]` — a real two-process `torchrun` job was launched on this machine
(gloo, CPU) and each rank reported itself:

```json
{"rank": 0, "local_rank": 0, "world_size": 2, "backend": "gloo", "is_primary": true,  "per_rank_batch_size": 16, "global_batch_size": 32}
{"rank": 1, "local_rank": 1, "world_size": 2, "backend": "gloo", "is_primary": false, "per_rank_batch_size": 16, "global_batch_size": 32}
```

That establishes the launch contract, the rank mapping, one-primary, and that
the derived per-rank batch reaches the workers. It establishes **nothing**
about any model's training under DDP.

`[STATIC]` — everything about NCCL, real GPUs, and model correctness under
DDP. This environment has `torch.cuda.device_count() == 0`.

## 8. Server acceptance still required

1. **2-GPU smoke.** CPTransformer, MATR, 1 epoch, 2 GPUs. Confirm with
   `nvidia-smi` that two workers occupy the two selected GPUs, that the logs
   show `world_size = 2` with ranks 0 and 1, and that the run produces
   exactly one predictions artifact and one metric set with the manifest at
   COMPLETED.
2. **Cancel.** Start a longer 2-GPU job and cancel it from the web UI.
   Confirm with `ps` that every torchrun worker is gone. Cancellation already
   signals the run's own process group (`core/run_manager.py`), which is what
   makes this work for a process tree; it is untested against torchrun on
   real GPUs.
3. **Resume under DDP.** Train to a resumable checkpoint, then resume to a
   larger total-epoch target. Confirm it starts at the stored epoch + 1, not
   at 0, and that the lineage is in the manifest.
4. **Equivalence.** One controlled small experiment, 1 GPU vs 2 GPU: same
   split, same total sample coverage, same intended effective batch, same
   output schema, comparable loss trajectory. Bitwise identity is **not**
   expected — reduction order across ranks differs — and DDP must not be
   claimed to change model quality either way.

## 9. Not implemented

* training through the distributed runner: it initialises the group, maps
  ranks, builds the sampler, wraps DDP and exits cleanly, but its `main()`
  deliberately refuses with a message rather than training, because wiring it
  to the pipeline needs a model whose DDP-safety has been established (§6);
* multi-node: `--nnodes=1` and `--standalone` only;
* BatLiNet DDP, for the scientific reason in §6;
* `DataParallel` anywhere, as a shortcut or otherwise.
