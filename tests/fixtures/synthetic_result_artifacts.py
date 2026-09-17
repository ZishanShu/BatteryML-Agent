# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""Synthetic saved-artifact payloads for the result-schema tests.

Per ``CLAUDE.md`` §7 these are test fixtures and nothing else: the
``synthetic_`` prefix is on the module and on every cell id, they are built in
memory rather than written under ``runs/`` or ``data/``, and nothing can pick
them up as a result. A fixture here may never reach ``metrics.csv`` or a
report.

Each builder returns the **shape ``Pipeline.evaluate`` writes** — a dict with
``prediction``, ``scores``, ``data``, ``seed``, and, for a run whose rows are
not one-per-cell, ``sample_index`` and ``task_metadata``. What varies between
them is only what the schema layer is allowed to read: the recorded task, and
the per-row identity.

The numbers are arbitrary and mean nothing. They exist so the arrays have a
length and so a residual can be computed; no builder here is a model, and no
value here is a measurement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Mirrors ``batteryml.data.sample_index.WHOLE_CELL_CYCLE`` — the cycle index
#: recorded for a row that is about a cell rather than a cycle.
WHOLE_CELL_CYCLE = -1

TASK_RUL = "RUL prediction"
TASK_SOH = "SOH estimation"


class _TestData:
    """The saved ``DataBundle.test_data``, as much as the views touch."""

    def __init__(self, label: Sequence[float]) -> None:
        self.label = list(label)


class _Bundle:
    def __init__(self, label: Sequence[float],
                 transformation: Any = None) -> None:
        self.test_data = _TestData(label)
        self.label_transformation = transformation


def _payload(
    *,
    prediction: Any,
    observed: Optional[Sequence[float]],
    sample_index: Optional[List[Dict[str, Any]]] = None,
    task_metadata: Optional[Dict[str, Any]] = None,
    scores: Optional[Dict[str, float]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prediction": prediction,
        "scores": dict(scores or {"MAE": 1.0, "RMSE": 2.0}),
        "seed": seed,
    }
    if observed is not None:
        payload["data"] = _Bundle(observed)
    if sample_index is not None:
        payload["sample_index"] = sample_index
    if task_metadata is not None:
        payload["task_metadata"] = task_metadata
    return payload


def _records(pairs: Sequence[Tuple[str, int]]) -> List[Dict[str, Any]]:
    return [{"cell_id": cell, "cycle_index": cycle} for cell, cycle in pairs]


# --- CASE 1: scalar RUL, one estimate per cell -----------------------------

def scalar_rul_payload(n_cells: int = 10) -> Dict[str, Any]:
    """Ten cells, ten lifetime predictions.

    Carries the index a whole-cell run really writes:
    ``SampleIndex.for_whole_cells`` gives every row ``cycle_index = -1``. That
    sentinel is the thing that stops "a sample_index exists" from being read
    as "this result is per-cycle" — which is exactly the mistake that would
    put a cycle curve on a lifetime result.
    """
    cells = [f"synthetic_cell_{i:02d}" for i in range(n_cells)]
    return _payload(
        prediction=[900.0 - 30.0 * i for i in range(n_cells)],
        observed=[880.0 - 30.0 * i for i in range(n_cells)],
        sample_index=_records([(cell, WHOLE_CELL_CYCLE) for cell in cells]),
        task_metadata={
            "task": TASK_RUL,
            "label_annotator": "RULLabelAnnotator",
            "feature_extractor": "VarianceModelFeatureExtractor",
        },
    )


def scalar_rul_payload_without_metadata(n_cells: int = 10) -> Dict[str, Any]:
    """The older artifact shape: no index and no recorded task at all.

    Read as scalar RUL, which is the contract that predates
    ``task_metadata`` and must keep holding for artifacts already on disk.
    """
    return _payload(
        prediction=[900.0 - 30.0 * i for i in range(n_cells)],
        observed=[880.0 - 30.0 * i for i in range(n_cells)],
    )


# --- CASE 2: cycle-wise RUL, several estimates per cell --------------------

#: Two cells, three observed cycles each, deliberately stored **out of cycle
#: order** for cell_B so that "sorted ascending within each cell" cannot pass
#: by accident.
CYCLEWISE_PAIRS: Tuple[Tuple[str, int], ...] = (
    ("synthetic_cell_A", 50),
    ("synthetic_cell_A", 100),
    ("synthetic_cell_A", 150),
    ("synthetic_cell_B", 150),
    ("synthetic_cell_B", 50),
    ("synthetic_cell_B", 100),
)

#: True remaining life at each of those cycles, in the row order above.
CYCLEWISE_OBSERVED: Tuple[float, ...] = (700.0, 650.0, 600.0, 540.0, 640.0, 590.0)
#: Estimated remaining life, likewise in row order.
CYCLEWISE_PREDICTED: Tuple[float, ...] = (690.0, 640.0, 575.0, 560.0, 655.0, 600.0)


def cyclewise_rul_payload() -> Dict[str, Any]:
    """Repeated RUL estimates for the same cell at several observed cycles.

    The only shape for which a RUL-vs-cycle curve is a real measurement.
    """
    return _payload(
        prediction=list(CYCLEWISE_PREDICTED),
        observed=list(CYCLEWISE_OBSERVED),
        sample_index=_records(CYCLEWISE_PAIRS),
        task_metadata={
            "task": TASK_RUL,
            "label_annotator": "RULLabelAnnotator",
            "feature_extractor": "VoltageCapacityMatrixFeatureExtractor",
        },
    )


# --- CASE 3: repeated rows, no cycle identity ------------------------------

def repeated_rows_without_cycle_identity_payload() -> Dict[str, Any]:
    """Six rows over two cells — but every row marked whole-cell.

    More rows than cells, and cell ids that repeat: both of the shape-only
    tests D4 forbids would call this cycle-wise. Nothing here says at which
    cycle any estimate was made, so no curve can be drawn from it.
    """
    cells = ["synthetic_cell_A"] * 3 + ["synthetic_cell_B"] * 3
    return _payload(
        prediction=list(CYCLEWISE_PREDICTED),
        observed=list(CYCLEWISE_OBSERVED),
        sample_index=_records([(cell, WHOLE_CELL_CYCLE) for cell in cells]),
        task_metadata={"task": TASK_RUL, "label_annotator": "RULLabelAnnotator"},
    )


def single_cycle_per_cell_payload() -> Dict[str, Any]:
    """Real cycle indices, but only one per cell.

    One point per cell is not a curve: there is no trend to show and joining
    two cells' single points would draw a line between different batteries.
    """
    return _payload(
        prediction=[690.0, 560.0],
        observed=[700.0, 540.0],
        sample_index=_records([("synthetic_cell_A", 50), ("synthetic_cell_B", 150)]),
        task_metadata={"task": TASK_RUL, "label_annotator": "RULLabelAnnotator"},
    )


# --- CASE 4: SOH ------------------------------------------------------------

SOH_PAIRS: Tuple[Tuple[str, int], ...] = (
    ("synthetic_cell_A", 10),
    ("synthetic_cell_A", 15),
    ("synthetic_cell_A", 20),
    ("synthetic_cell_B", 10),
    ("synthetic_cell_B", 15),
    ("synthetic_cell_B", 20),
)
SOH_OBSERVED: Tuple[float, ...] = (1.00, 0.97, 0.94, 0.99, 0.95, 0.90)
SOH_PREDICTED: Tuple[float, ...] = (0.98, 0.99, 0.92, 1.02, 0.94, 0.88)


def soh_cyclewise_payload() -> Dict[str, Any]:
    """A cycle-level SOH artifact: the shape the verified MATR route writes.

    Structurally identical to :func:`cyclewise_rul_payload` — repeated cells,
    several real cycles each. Only the recorded task tells them apart, which
    is the whole reason the task is recorded.
    """
    return _payload(
        prediction=list(SOH_PREDICTED),
        observed=list(SOH_OBSERVED),
        sample_index=_records(SOH_PAIRS),
        task_metadata={
            "task": TASK_SOH,
            "label_annotator": "CycleSOHLabelAnnotator",
            "reference_capacity_mode": "first_valid_discharge_capacity",
            "min_cycle": 10,
            "max_cycle": None,
            "cycle_stride": 5,
            "feature_extractor": "SOHVoltageCurveFeatureExtractor",
        },
    )


def soh_cyclewise_payload_without_declared_task() -> Dict[str, Any]:
    """The same, identified only by its label annotator.

    A config that declares no ``task:`` still records the annotator, and the
    annotator is what the run trained against.
    """
    payload = soh_cyclewise_payload()
    payload["task_metadata"].pop("task")
    return payload


def soh_scalar_payload() -> Dict[str, Any]:
    """One SOH number per cell, from the whole-cell ``SOHLabelAnnotator``."""
    cells = [f"synthetic_cell_{i:02d}" for i in range(4)]
    return _payload(
        prediction=[0.98, 0.95, 0.91, 0.88],
        observed=[0.97, 0.96, 0.92, 0.90],
        sample_index=_records([(cell, WHOLE_CELL_CYCLE) for cell in cells]),
        task_metadata={"task": TASK_SOH, "label_annotator": "SOHLabelAnnotator"},
    )


# --- CASE 5: trajectory -----------------------------------------------------

def trajectory_payload() -> Dict[str, Any]:
    """A sequence per cell: a forecast future, not an estimate at a cycle."""
    return _payload(
        prediction=[
            [1.00, 0.98, 0.96, 0.94],
            [1.00, 0.97, 0.93, 0.89],
        ],
        observed=[
            [1.00, 0.99, 0.97, 0.95],
            [1.00, 0.96, 0.92, 0.88],
        ],
        task_metadata={"task": TASK_RUL, "label_annotator": "RULLabelAnnotator"},
    )


def trajectory_payload_with_cycle_index() -> Dict[str, Any]:
    """A trajectory that also carries a cycle-level index and RUL semantics.

    The adversarial case for D4: every piece of evidence a shape-only
    classifier looks at points at "cycle-wise RUL", and the predictions are
    still sequences. A sequence per row is never a scalar estimate at a
    cycle, so this must stay a trajectory.
    """
    payload = trajectory_payload()
    payload["sample_index"] = _records([
        ("synthetic_cell_A", 50), ("synthetic_cell_A", 100),
    ])
    return payload
