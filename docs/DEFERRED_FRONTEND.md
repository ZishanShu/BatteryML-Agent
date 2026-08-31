# Deferred frontend fixes

`batteryml_web/app.py` and every Streamlit page are frozen. Defects found in
them during backend work are recorded here — file, line, symptom, and the
change needed later — and left in place.

Each entry states how it was established. Nothing here was observed by running
Streamlit; this environment cannot (`CLAUDE.md` §0).

---

## DF-01 — `ConfigEntry` has no `.model` attribute (two call sites, both `AttributeError`)

| Field | Value |
| --- | --- |
| File | `batteryml_web/app.py` |
| Lines | 841 and 1075 |
| Found while | Fixing the config registry (item 1). |
| Basis | `[STATIC]` — read at `batteryml_web/app.py:841`, `:1075` and `batteryml_web/core/config_registry.py:100-107`. |

### Symptom

```python
# app.py:841, inside _discover_run_configs()
config_map.setdefault((entry.dataset, entry.task, entry.model), []).append(entry.config_path)

# app.py:1075, inside the Run tab
available_models = sorted({entry.model for entry in get_config_registry(str(REPO_ROOT)).get_entries(ds, task_choice)})
```

`ConfigEntry` is a frozen dataclass whose fields are `dataset`, `task`,
`model_key`, `model_display`, `split_preset`, `split_label`, `config_path`,
`metadata` (`batteryml_web/core/config_registry.py:100-107`). There is no
`model` field, so both expressions raise
`AttributeError: 'ConfigEntry' object has no attribute 'model'` the first time
they are reached.

This is not caused by the registry fix — both lines predate it and would have
failed identically before. The registry fix makes them reachable for 24 more
configs, but they were already broken for the other 90.

### Change needed later

Decide which of the two fields each call site wants, then use it:

* `:841` builds a lookup key. Use `entry.model_key` — the stable identifier
  (`xgb`, `discharge_model`), which is what a key should be built from.
* `:1075` populates a `st.selectbox` the user reads. Use `entry.model_display`
  (`XGBoost — Extreme Gradient Boosting`) for the label, and carry
  `entry.model_key` as the value, so the stored
  `st.session_state["selected_model"]` stays a stable key rather than a
  display string.

Note that `:1075` writes the selection into `st.session_state["selected_model"]`
and `:841` builds a key tuple; if the two disagree about which field they use,
the lookup at `:841` will silently miss. Fix both together.

---

## DF-02 — `_infer_dataset` matches "matr" inside "VoltageCapacityMatrixFeatureExtractor"

| Field | Value |
| --- | --- |
| File | `batteryml_web/core/config_registry.py` (**not** frozen), surfaced through `batteryml_web/app.py:1075` |
| Lines | `config_registry.py:148-157` |
| Found while | Fixing `_infer_model` (item 1). |
| Basis | `[VERIFIED]` — reproduced by building the registry in this session and reading back the `dataset` field of every entry. |

### Symptom

`_infer_dataset` searches for a dataset name as a substring of
`f"{path} {path.name} {text}".lower()`. Every config that declares

```yaml
feature:
    name: 'VoltageCapacityMatrixFeatureExtractor'
```

contains the substring `matr` inside `Matrix`. Since `MATR` is first in the
search order, **every** such config is reported as dataset `MATR`, including
`cruh`, `crush`, `hust`, `snl` and `mix`, which point at CALCE / RWTH / UL_PUR /
HNEI / SNL / HUST.

The consequence in the UI: the dataset selector at `app.py:1067` filters
`get_entries(ds, task_choice)` by this field, so choosing `HUST` shows nothing
while choosing `MATR` shows configs that will train on data the user did not
select.

### Why it was not fixed now

It is a genuine backend defect in a non-frozen file, but fixing it changes which
configs appear under which dataset in the UI — a visible behaviour change
outside the scope of item 1, which was specifically the model-key mismatch at
`config_registry.py:26-28`. Recorded here so the two are decided together.

### Change needed later

Resolve the dataset the same way `_infer_model` now resolves the model: from
declared structure rather than raw text. `train_test_split.cell_data_path` names
the processed directories explicitly (a string or a list), so
`Path(p).name for p in cell_data_path` gives the exact dataset set with no
substring matching. A config that points at several directories has several
datasets, which the current single-valued `ConfigEntry.dataset` field cannot
express — that is the design decision this needs.

`SPLIT-01` (`scripts/verify/check_split_requirements.py`) already derives the
correct per-preset dataset sets this way and can be reused.

---

## DF-03 — `agent/troubleshoot_agent.py` has the same two import defects that were repaired in its siblings

| Field | Value |
| --- | --- |
| File | `batteryml_web/agent/troubleshoot_agent.py` (**not** frozen) |
| Lines | 7, 8, 9 |
| Found while | Repairing `schema_agent.py` and `result_agent.py` (item 4). |
| Basis | `[VERIFIED]` — `import batteryml_web.agent.troubleshoot_agent` raises `ModuleNotFoundError: No module named 'core'`; observed in this session. That is line 7 failing first; the `LLMClient` defect on line 9 is `[STATIC]`, established by reading `agent/llm_client.py`, which defines no such name. |

### Symptom

```python
from core.config_validator import load_and_validate_config_file   # line 7
from core.schema_validator import validate_schema_mapping         # line 8
from .llm_client import LLMClient                                 # line 9
```

* Line 9 imports `LLMClient`, which `agent/llm_client.py` does not define — it
  exposes the functions `run_llm_prompt`, `test_llm_connection` and
  `save_llm_output`. Same defect as `schema_agent.py` and `result_agent.py`.
* Lines 7-8 are absolute imports of `core.*`, which resolve only if
  `batteryml_web/` is itself on `sys.path`. The live layout puts the repo root
  on `sys.path` (`app.py` imports `batteryml_web.agent.llm_client`), so they
  fail. Same defect as `agent/prompts.py`, which was fixed because
  `schema_agent` imports it.

Nothing imports `troubleshoot_agent` today, so it fails only when something
starts to.

### Why it was not fixed now

It is a **fifth** broken module; item 4 named four. Error diagnosis is agent
layer under `CLAUDE.md` §3, so by the same reasoning applied to `schema_agent`
and `result_agent` it should be repaired rather than deleted — but that is your
call, not an assumption to act on.

### Change needed later

Change lines 7-8 to `from ..core.config_validator import ...` /
`from ..core.schema_validator import ...`, and rewrite the `LLMClient` usage
against `run_llm_prompt(prompt, *, base_url, api_key, model, system_prompt,
timeout)` exactly as `agent/schema_agent.py` now does. `diagnose_failure` and
`run_validator_after_fix` otherwise look structurally sound `[STATIC]`.
