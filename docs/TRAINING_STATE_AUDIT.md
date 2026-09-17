# Audit: training state, resume, and DDP readiness

Milestones A1 and B1 require this audit before any implementation. Every row
is `[STATIC]` — read from the source at the cited lines — unless marked
otherwise. Nothing here was observed by training a model: this environment
has no GPU (`torch.cuda.device_count() == 0`), no `data/`, and no
`third_party/`.

---

## 0. What is present in this environment

| Component | Present | Consequence |
| --- | --- | --- |
| `batteryml/` core | yes | auditable and modifiable |
| `batteryml_web/models/batlinet.py` | yes | auditable (1060 lines) |
| `batteryml_web/models/cptransformer.py` | yes (adapter only) | the adapter is auditable; the **model** is not |
| `third_party/BatteryLife/models/CPTransformer.py` | **no** | the network's own source cannot be read here |
| `data/` | **no** | no training can be run |
| CUDA devices | **0** | no GPU execution, no NCCL |
| `torchrun`, gloo | yes | distributed *orchestration* is genuinely testable on CPU |

`batteryml_web/models/cptransformer.py:22-27` raises `FileNotFoundError` at
import when the BatteryLife source is absent, which is why 58 tests skip
here. Any claim about CPTransformer's internals would therefore be a guess,
and is not made.

---

## 1. Training state per model (A1)

### `NNModel.fit` — `batteryml/models/nn_model.py:47-86`

The base class every neural model inherits.

| State | Held where | Persisted by `dump_checkpoint`? |
| --- | --- | --- |
| model `state_dict` | the module | **yes** |
| optimizer state | `optimizer` — a **local variable** created at `:56` inside `fit()` | **no** |
| LR scheduler | **does not exist** — no scheduler is constructed anywhere | n/a |
| AMP / `GradScaler` | **does not exist** — no autocast, no scaler | n/a |
| current epoch | `epoch`, the loop variable at `:61` | **no** |
| best metric / early stopping | **does not exist** — every epoch runs; there is no best-state tracking | n/a |
| Python `random` state | global; seeded per worker by `seed_worker` (`:19-22`) | **no** |
| NumPy RNG state | global; same | **no** |
| torch CPU RNG state | global; drives `DataLoader(shuffle=True)` at `:51-53` | **no** |
| torch CUDA RNG state | global, per device | **no** |
| dataset / split identity | `DataBundle` passed in; not recorded in the checkpoint | **no** |

`dump_checkpoint` is `torch.save(self.state_dict(), path)` (`:106-107`) and
`load_checkpoint` is `self.load_state_dict(torch.load(path))` (`:109-110`).

**Verdict: the existing `.ckpt` is a weights-only artifact.** Restoring it
and continuing gives a fresh Adam (moment estimates lost), an epoch counter
restarting at 0, and a different data order. It is a legitimate warm start;
it is not a resume.

One further detail with a real consequence: at `:64-66`, a non-finite loss
triggers `reset_parameters(self)` **and rebuilds the optimizer**. Any resume
design must treat the optimizer as something that can be replaced mid-run,
not as an object whose identity persists for the whole of training.

### `BatLiNetRULPredictor.fit` — `batteryml_web/models/batlinet.py:813-915`

Does not use `DataLoader` at all.

| State | Held where | Persisted? |
| --- | --- | --- |
| network `state_dict` | `self.network` | yes (`:1036`) |
| optimizer | local, created at `:837` | no |
| scheduler / scaler | do not exist | n/a |
| epoch | loop variable `:853` | no |
| RNG | `torch.randperm(n, device=device)` at `:860` — the epoch's data order comes straight from the global torch RNG | no |
| support-set sampling | `self._training_support(features, labels, indices)` at `:884` | no |

### sklearn / XGBoost models — `batteryml/models/sklearn_model.py`

`load_checkpoint` exists but these estimators have no mid-training state at
all: `fit` is a single call with no epoch loop. "Resume" has no meaning for
them, and inventing one would be a fiction.

### Resume capability table

| Model family | Resume supportable? | Reason |
| --- | --- | --- |
| `NNModel` subclasses (CNN, LSTM, MLP, Transformer…) | **yes, once the loop persists state** | epoch loop + Adam + torch RNG are all capturable |
| BatLiNet | **yes in principle**, not in this milestone | same shape, but its own `fit` must be changed too and its support sampling adds RNG surface |
| CPTransformer | **unknown — source absent** | the adapter subclasses `NNModel`, but whether the network holds extra training state cannot be read here |
| sklearn / XGBoost | **no** | no iterative training state exists to resume |

---

## 2. DDP readiness (B1)

| Model | DDP-safe? | Reason | Required changes |
| --- | --- | --- | --- |
| `NNModel` generic loop | **partially** | uses a real `DataLoader`, so a `DistributedSampler` fits naturally | sampler, `set_epoch`, DDP wrap, rank-0 side effects |
| CPTransformer | **unverifiable here** | the network source is absent; DDP-safety depends on its forward/loss, which cannot be read | audit on the server, then wire |
| BatLiNet | **no** | see below | a distributed redesign of support sampling |
| sklearn / XGBoost | **no** | not torch models | n/a |

### Why BatLiNet is not DDP-safe as written

`_training_support(features, labels, indices)` (`:884`) draws a support set
for each batch **from the full training tensor**. Under DDP each rank would
hold only its shard, so either:

* each rank samples support cells only from its own shard — which changes
  which inter-cell comparisons the model ever sees, and therefore changes the
  objective; or
* every rank keeps the full feature tensor — which removes the memory
  benefit that motivated multi-GPU in the first place, and still changes the
  sampling distribution per step.

Both alter the science. BatLiNet therefore stays
`supports_multi_gpu = false` with that reason recorded, per B3.

### Data placement

`DataBundle.to(device)` (`batteryml/data/databundle.py:67-74`) moves the
**entire** train and test tensors onto one device. `Pipeline` calls it once.
This is why a large dataset does not fit on one GPU — and why naive DDP that
replicates the whole bundle on every rank would not help. A correct DDP path
must feed rank-local minibatches from CPU-resident data via a
`DistributedSampler`, not move the whole bundle per rank.

### Effective batch size

`NNModel.fit` uses `self.train_batch_size` as the **per-process** batch size
for its `DataLoader`. Under DDP with `world_size = N` and no policy, the
global batch becomes `N × train_batch_size` — a silent change to the
optimization, since Adam's updates would be computed over N times as many
samples per step. The policy chosen in Milestone B is recorded in
`docs/DDP_CONTRACT.md`.

---

## 3. What this audit licenses

* **Milestone A** can be implemented and genuinely tested here, on a tiny
  deterministic model, including an A/B equivalence check. It requires a
  change to `batteryml/models/nn_model.py`, recorded in
  `docs/UPSTREAM_PATCHES.md` as `CLAUDE.md` §4 requires.
* **Milestone B**'s launcher contract, distributed runner, rank-0 rules,
  batch-size policy, capability gating and cancellation can be implemented
  and tested on CPU/gloo with a real `torchrun`.
* **Milestone B's CPTransformer model-side DDP cannot be completed or
  verified in this environment**, because the network's source is not here
  and there are no GPUs. That work, and the 2-GPU proof B2 asks for, belong
  on the server.
