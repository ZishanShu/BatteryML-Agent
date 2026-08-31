# Upstream BatteryML patches

Per `CLAUDE.md` §4, `batteryml/` is upstream core and is not modified. Every
unavoidable change to it is recorded here with rationale and a minimal diff.

Upstream reference: <https://github.com/microsoft/BatteryML> (branch `main`).

---

## UP-01 — `TransformerRULPredictor.forward` printed the feature shape on every forward pass

| Field | Value |
| --- | --- |
| File | `batteryml/models/rul_predictors/transformer.py` |
| Line (before patch) | 35 |
| Upstream status | Present in `microsoft/BatteryML@main`. Confirmed by fetching `https://raw.githubusercontent.com/microsoft/BatteryML/main/batteryml/models/rul_predictors/transformer.py` on 2026-08-31; that file is byte-identical to this repository's pre-patch copy, `print(feature.shape)` included. `[VERIFIED]` — the fetch and comparison were performed and observed. |
| Approved by | User instruction, this session. |

### Symptom

`forward()` is called once per training batch and once per evaluation batch.
Every call wrote a bare `torch.Size([...])` line to stdout. The transformer
configs declare `epochs: 1000` with `batch_size: 128`
(`configs/baselines/nn_models/transformer/matr_1.yaml`), so a single run emits
on the order of 10^4 unlabelled shape lines into the same stream that the
metrics parser reads.

`bin/batteryml.py::run` prints the evaluation scores dict to the same stdout, so
this noise is interleaved with the only machine-readable result the CLI emits.

### Why not simply delete the line

The shape is genuinely useful when debugging the 3-D/4-D feature branch
immediately below it (`feature.ndim == 4` vs `== 3`), which is the code path
that differs between feature extractors. The line is therefore demoted to
`logging.DEBUG` rather than removed, so it can be turned back on without
editing core.

### Minimal diff

```diff
--- a/batteryml/models/rul_predictors/transformer.py
+++ b/batteryml/models/rul_predictors/transformer.py
@@ -1,11 +1,15 @@
 # Licensed under the MIT License.
 # Copyright (c) Microsoft Corporation.

+import logging
+
 import torch
 import torch.nn as nn

 from batteryml.builders import MODELS
 from batteryml.models.nn_model import NNModel

+logger = logging.getLogger(__name__)
+

 @MODELS.register()
 class TransformerRULPredictor(NNModel):
@@ -31,7 +35,8 @@ class TransformerRULPredictor(NNModel):
     def forward(self,
                 feature: torch.Tensor,
                 label: torch.Tensor,
                 return_loss: bool = False):
-        print(feature.shape)
+        logger.debug('TransformerRULPredictor input shape: %s',
+                     tuple(feature.shape))
         if feature.ndim == 4:
```

### Behaviour after the patch

`batteryml/preprocess/__init__.py` calls `logging.basicConfig(level=logging.INFO)`
at import time, so the root logger sits at `INFO` and this record is suppressed
by default. `[STATIC]` — `batteryml/preprocess/__init__.py:14-16`.

To re-enable it for a debugging run, raise the level for this logger only:

```python
import logging
logging.getLogger('batteryml.models.rul_predictors.transformer').setLevel(logging.DEBUG)
```

No numeric behaviour changes: the statement has no effect on `feature`, on the
loss, or on any prediction. `[STATIC]`

### If this is ever upstreamed

The change is self-contained and carries no repository-specific assumptions; it
could be submitted to `microsoft/BatteryML` unmodified.
