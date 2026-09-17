"""Prediction artifacts -> validated arrays, with the three alignment gates.

Nothing here plots and nothing here writes. The module exists because three
things about a BatteryML prediction artifact are easy to get wrong in a way no
reader of the resulting figure could detect:

1. **The label space.** ``predictions_seed_*.pkl`` stores the prediction and the
   test labels in the *transformed* space. The demo config stacks
   ``LogScale -> ZScore`` (``configs/baselines/sklearn/xgb/matr_1.yaml:16-21``),
   so the raw tensors are z-scores of order 1, while the recorded RMSE is a
   number of cycles of order 100. ``DataBundle.evaluate`` inverse-transforms
   before scoring (``batteryml/data/databundle.py:88-92``); a figure that skips
   that step plots axes which silently disagree with ``metrics.csv``.
   :func:`load_prediction_artifact` inverse-transforms, then calls
   :func:`verify_recorded_scores` to prove it reproduced the recorded numbers.

2. **The NaN-label mask.** ``Task.build`` drops cells whose label is NaN from
   the feature and label tensors (``batteryml/task.py:62-68``) but keeps the
   full cell list on the task (``:53-54``). So ``test_cells[i]`` is *not*
   row ``i`` of ``test_data.label`` whenever any cell was dropped. Any per-cell
   figure built on the naive assumption mislabels every point.
   :func:`resolve_test_cell_ids` re-derives the mask instead of assuming it.

3. **Cell identity is not always present.** The CLI path
   (``batteryml/pipeline.py:113-122``) pickles ``prediction``, ``scores``,
   ``data`` and ``seed`` — no cell IDs. Only the manual path
   (``scripts/reproduce_batteryml/manual_train_eval.py:85-88``) adds
   ``raw_data``. When identity cannot be established, the per-cell figure is
   skipped with a stated reason; it is never drawn against guessed labels.

Every failure raises :class:`VizDataError` naming the offending file and the
corrective action (``CLAUDE.md`` §6). No bare excepts, no defaults substituted
for missing inputs.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Config fields BatteryML reads from a run config. Mirrors
# ``batteryml/pipeline.py:141-148``; duplicated rather than imported so that
# reading a config does not drag in the model registry and its optional
# third-party backends.
CONFIG_FIELDS = (
    "model",
    "train_test_split",
    "feature",
    "label",
    "feature_transformation",
    "label_transformation",
)

# Keys under which the two writers store the test prediction.
PREDICTION_KEYS = ("prediction", "test_prediction")

# Metrics BatteryML's evaluator implements (``databundle.py:100``). Only these
# can be cross-checked; anything else in ``scores`` is carried through untouched.
CHECKABLE_METRICS = ("RMSE", "MAE", "MAPE")

# How a label annotator's output should be described on an axis.
LABEL_UNITS = {
    "RULLabelAnnotator": "cycles",
    # A ratio, not a percentage, and not "of nominal capacity": the reference
    # is whatever the run's reference_capacity_mode says, which the artifact
    # records in task_metadata. Naming nominal capacity here would have
    # asserted a convention the run may not have used.
    "SOHLabelAnnotator": "SOH (ratio)",
    "CycleSOHLabelAnnotator": "SOH (ratio)",
}


class VizDataError(RuntimeError):
    """A prediction artifact, config or cell list could not be used as given.

    Always carries the offending path or field and the action that would fix it.
    """


@dataclass(frozen=True)
class CellIdResolution:
    """The outcome of trying to name the cell behind each prediction.

    Attributes:
        cell_ids: one ID per prediction, in prediction order, or ``None`` when
            identity could not be established.
        source: how the IDs were obtained — ``artifact_raw_data``,
            ``artifact_raw_data+label_mask``, ``splitter_paths``,
            ``splitter_paths+label_mask``, or ``unavailable``.
        n_cells_total: cells in the test split before the NaN-label mask.
        n_dropped_nan_label: cells the mask removed, or ``None`` if not derived.
        alignment_verified: True only when the labels recomputed from the cells
            were compared against the artifact's own labels and matched. False
            means the alignment is inferred from a count match alone.
        reason: one sentence a human can act on. Explains why ``cell_ids`` is
            ``None``, or records how the mapping was established.
    """

    cell_ids: list[str] | None
    source: str
    n_cells_total: int | None
    n_dropped_nan_label: int | None
    alignment_verified: bool
    reason: str


@dataclass(frozen=True, eq=False)
class PredictionArtifact:
    """One evaluated BatteryML run, in the label space the metrics are in.

    Attributes:
        source_path: the ``.pkl`` this was read from.
        source_sha256: its content hash, recorded so a figure can be traced back
            to the exact artifact that produced it.
        seed: the seed recorded by BatteryML, or ``None`` if absent.
        y_true: test labels, inverse-transformed. Shape ``(n,)``, float64.
        y_pred: predictions, inverse-transformed. Shape ``(n,)``, float64.
        scores: the evaluator's own metric values, verbatim.
        score_check: per metric, the recorded value, the value recomputed from
            ``y_true``/``y_pred`` by BatteryML's evaluator, and their relative
            difference. Evidence that the inverse transform was right.
        label_transformations: class names of the transformations that were
            inverted, outermost first. Empty when the config declared none.
        cells: how cell identity was resolved.
    """

    source_path: Path
    source_sha256: str
    seed: int | None
    y_true: np.ndarray
    y_pred: np.ndarray
    scores: dict[str, float]
    score_check: dict[str, dict[str, float]]
    label_transformations: list[str]
    cells: CellIdResolution
    #: For a cycle-level task, one ``{"cell_id", "cycle_index"}`` per
    #: prediction in row order, written by ``Pipeline.evaluate`` (UP-06).
    #: ``None`` for a whole-cell task such as RUL, where the row order is the
    #: cell order and there is no cycle to record. This is the *recorded*
    #: identity, not the one ``cells`` infers from the splitter.
    sample_index: list[dict[str, Any]] | None = None
    #: The label conventions the run used — ``reference_capacity_mode`` and
    #: the sample policy for SOH. Empty when the artifact records none.
    task_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_cycle_level(self) -> bool:
        """Whether each prediction names an observed cycle, not just a cell."""
        return bool(self.sample_index)

    @property
    def n_predictions(self) -> int:
        """Number of test points in the artifact."""
        return int(self.y_true.shape[0])

    @property
    def residuals(self) -> np.ndarray:
        """``y_pred - y_true``. A display quantity, never a reported metric."""
        return self.y_pred - self.y_true


def _sample_index_of(payload: Any, source: Path, n_rows: int) -> list[dict[str, Any]] | None:
    """The recorded ``(cell_id, cycle_index)`` per prediction, or None.

    Validated rather than trusted: a sample index whose length disagrees with
    the predictions would silently mislabel every point of an SOH-vs-cycle
    plot, which is exactly the class of error this module exists to prevent.
    A whole-cell artifact records none and gets ``None`` — that is normal, not
    a fault.
    """
    if not isinstance(payload, dict):
        return None
    records = payload.get("sample_index")
    if records is None:
        return None
    if not isinstance(records, (list, tuple)):
        raise VizDataError(
            f"{source}: 'sample_index' is {type(records).__name__}, expected a "
            "list of {{cell_id, cycle_index}} records. The artifact was not "
            "written by this platform's evaluate step."
        )
    if len(records) != n_rows:
        raise VizDataError(
            f"{source}: 'sample_index' has {len(records)} record(s) but the "
            f"artifact holds {n_rows} prediction(s). Every prediction must "
            "name the cell and cycle it was made for; a mismatch would "
            "mislabel every point of a per-cycle view."
        )
    out: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        if not isinstance(record, dict) or "cell_id" not in record or "cycle_index" not in record:
            raise VizDataError(
                f"{source}: sample_index[{position}] = {record!r} is missing "
                "'cell_id' or 'cycle_index'."
            )
        out.append({
            "cell_id": str(record["cell_id"]),
            "cycle_index": int(record["cycle_index"]),
        })
    return out


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks so a large artifact fits in memory.

    Args:
        path: the file to hash.

    Returns:
        The 64-character lowercase hex digest.

    Raises:
        VizDataError: if the path does not exist or is not a file.
    """
    path = Path(path)
    if not path.is_file():
        raise VizDataError(
            f"cannot hash {path}: it does not exist or is not a regular file."
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_registries() -> None:
    """Import the modules whose side effect is registering BatteryML classes.

    Building a label annotator or a splitter by name needs the registration
    decorators to have run. Mirrors the import block in
    ``scripts/reproduce_batteryml/manual_train_eval.py:17-23``, minus the model
    zoo, which nothing here builds.

    Raises:
        VizDataError: naming the module that failed and the likely cause.
    """
    modules = (
        "batteryml.data.transformation.log_scale",
        "batteryml.data.transformation.sequential",
        "batteryml.data.transformation.z_score",
        "batteryml.label",
        "batteryml.train_test_split",
    )
    import importlib

    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise VizDataError(
                f"cannot import {module}, which registers the classes a config names: {exc}. "
                "Run from the repository root with the BatteryML dependencies installed "
                "(`pip install -r requirements.txt`)."
            ) from exc


def read_config(config_path: Path) -> dict[str, Any]:
    """Read a BatteryML run config without touching the filesystem elsewhere.

    Deliberately does **not** call ``batteryml.pipeline.load_config``: that
    resolves and *creates* a workspace directory (``pipeline.py:174-175``).
    Reading a config in order to draw a figure must not create directories.

    Args:
        config_path: path to the ``.yaml`` run config.

    Returns:
        A dict with one entry per field in :data:`CONFIG_FIELDS`.

    Raises:
        VizDataError: if the file is missing or cannot be parsed.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise VizDataError(
            f"config {config_path} does not exist. Pass --config with the same YAML "
            "the run was launched with."
        )
    from batteryml.utils import import_config

    try:
        return import_config(config_path, list(CONFIG_FIELDS))
    except Exception as exc:
        raise VizDataError(
            f"cannot read config {config_path}: {type(exc).__name__}: {exc}. "
            "Check it parses as YAML and declares model / train_test_split / feature / label."
        ) from exc


def label_unit_from_config(config: dict[str, Any] | None) -> str:
    """Axis unit for the label the config annotates.

    Args:
        config: a config as returned by :func:`read_config`, or ``None``.

    Returns:
        ``"cycles"`` for RUL, an SOH description for SOH, and the neutral
        ``"label"`` when the annotator is unknown or no config was given. The
        neutral value is deliberate: a wrong unit on an axis is worse than none.
    """
    if not config:
        return "label"
    name = str((config.get("label") or {}).get("name", "")).strip()
    return LABEL_UNITS.get(name, "label")


def dataset_from_config(config: dict[str, Any] | None) -> str | None:
    """Dataset name from the processed directories the splitter is pointed at.

    Derived from ``train_test_split.cell_data_path`` rather than by matching
    dataset names against the config text — the substring approach reports every
    config using ``VoltageCapacityMatrixFeatureExtractor`` as MATR, because
    ``matr`` occurs inside ``Matrix`` (``docs/DEFERRED_FRONTEND.md`` DF-02).

    Args:
        config: a config as returned by :func:`read_config`, or ``None``.

    Returns:
        The directory name for a single-dataset config, the names joined by
        ``+`` for a pooled one, or ``None`` when no path is declared.
    """
    if not config:
        return None
    paths = (config.get("train_test_split") or {}).get("cell_data_path")
    if not paths:
        return None
    if isinstance(paths, str):
        paths = [paths]
    names = [Path(str(item)).name for item in paths if str(item).strip()]
    if not names:
        return None
    return "+".join(dict.fromkeys(names))


def _as_1d_float_tensor(value: Any, *, field: str, source: Path) -> torch.Tensor:
    """Coerce a prediction or label to a 1-D float32 CPU tensor.

    Args:
        value: a torch tensor, numpy array, or sequence of numbers.
        field: the artifact key it came from, for the error message.
        source: the artifact path, for the error message.

    Returns:
        A detached, contiguous 1-D float32 tensor on the CPU.

    Raises:
        VizDataError: if the value is not numeric or is not 1-D after squeezing
            a trailing singleton dimension.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(value))
    elif isinstance(value, (list, tuple)):
        try:
            tensor = torch.tensor(value)
        except Exception as exc:
            raise VizDataError(
                f"{source}: field {field!r} is a {type(value).__name__} that does not "
                f"convert to a numeric tensor: {exc}."
            ) from exc
    else:
        raise VizDataError(
            f"{source}: field {field!r} is a {type(value).__name__}; expected a torch "
            "tensor, a numpy array or a sequence of numbers."
        )

    tensor = tensor.reshape(-1).float()
    if tensor.numel() == 0:
        raise VizDataError(
            f"{source}: field {field!r} is empty. The run produced no test points; "
            "re-run the evaluation before drawing figures."
        )
    return tensor


def _transformation_chain(transformation: Any) -> list[str]:
    """Class names of a (possibly nested) label transformation, outermost first.

    Args:
        transformation: the ``label_transformation`` carried by the DataBundle,
            or ``None``.

    Returns:
        ``[]`` when there is no transformation; otherwise the class name, with a
        ``SequentialDataTransformation`` expanded into its members in the order
        they are applied forward.
    """
    if transformation is None:
        return []
    members = getattr(transformation, "transformations", None)
    if isinstance(members, (list, tuple)) and members:
        chain: list[str] = []
        for member in members:
            chain.extend(_transformation_chain(member))
        return chain
    return [type(transformation).__name__]


def verify_recorded_scores(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    scores: dict[str, float],
    *,
    tolerance: float,
    source: Path,
) -> dict[str, dict[str, float]]:
    """Prove the inverse-transformed arrays reproduce the recorded metrics.

    This is a **gate, not a measurement**. The numbers a figure displays are the
    recorded ones; this function only establishes that the arrays about to be
    plotted live in the same space those numbers were computed in. It calls
    BatteryML's own evaluator (``DataBundle._evaluate_score``) so the comparison
    is against the definition BatteryML used, not a reimplementation.

    Args:
        y_true: inverse-transformed labels, 1-D float tensor.
        y_pred: inverse-transformed predictions, same shape.
        scores: the artifact's recorded metric values.
        tolerance: maximum relative difference accepted per metric.
        source: artifact path, for the error message.

    Returns:
        Per checkable metric present in ``scores``: ``recorded``, ``recomputed``
        and ``relative_difference``.

    Raises:
        VizDataError: if any checkable metric disagrees by more than
            ``tolerance``. That means the label space is wrong, so every figure
            drawn from these arrays would contradict ``metrics.csv``.
    """
    from batteryml.data.databundle import DataBundle

    report: dict[str, dict[str, float]] = {}
    mismatched: list[str] = []
    for metric in CHECKABLE_METRICS:
        if metric not in scores:
            continue
        recorded = float(scores[metric])
        recomputed = float(DataBundle._evaluate_score(y_true, y_pred, metric))
        scale = max(abs(recorded), 1e-12)
        relative = abs(recomputed - recorded) / scale
        report[metric] = {
            "recorded": recorded,
            "recomputed": recomputed,
            "relative_difference": relative,
        }
        if relative > tolerance:
            mismatched.append(
                f"{metric}: recorded {recorded!r}, recomputed {recomputed!r} "
                f"(relative difference {relative:.3e})"
            )

    if mismatched:
        raise VizDataError(
            f"{source}: the inverse-transformed labels do not reproduce the recorded "
            f"scores — {'; '.join(mismatched)}. Every figure drawn from these arrays "
            "would disagree with metrics.csv, so none was written. The usual cause is "
            "that the artifact's DataBundle carries a different label_transformation "
            "than the run that produced the scores; confirm the artifact and the "
            "config belong to the same run. Raise --score-tolerance only if you have "
            "established the difference is float32 rounding."
        )
    if not report:
        raise VizDataError(
            f"{source}: 'scores' contains none of {', '.join(CHECKABLE_METRICS)}, so the "
            f"label space cannot be checked. Found: {sorted(scores)}. Re-run the "
            "evaluation with --metric RMSE,MAE,MAPE."
        )
    return report


def _cell_ids_of(cells: list[Any], *, source: Path) -> list[str]:
    """Read ``cell_id`` off each BatteryData, failing loudly on a blank one."""
    ids: list[str] = []
    for index, cell in enumerate(cells):
        cell_id = getattr(cell, "cell_id", None)
        if cell_id is None or not str(cell_id).strip():
            raise VizDataError(
                f"{source}: test cell at index {index} has no cell_id "
                f"(object of type {type(cell).__name__}). Per-cell figures need identity; "
                "re-run preprocessing for this dataset."
            )
        ids.append(str(cell_id))
    return ids


def _labels_for_cells(cells: list[Any], config: dict[str, Any], *, source: Path) -> np.ndarray:
    """Recompute the annotator's label for every cell, NaNs included.

    The NaN entries are the whole point: they are what ``Task.build`` masks out,
    and reproducing the mask is the only way to align predictions with cells.
    """
    _import_registries()
    from batteryml.builders import LABEL_ANNOTATORS

    label_config = config.get("label")
    if not label_config or not dict(label_config).get("name"):
        raise VizDataError(
            f"{source}: the config declares no 'label.name', so the NaN-label mask that "
            "Task.build applied cannot be reproduced and predictions cannot be matched "
            "to cells."
        )
    try:
        annotator = LABEL_ANNOTATORS.build(dict(label_config))
    except Exception as exc:
        raise VizDataError(
            f"cannot build the label annotator {dict(label_config).get('name')!r} named by "
            f"the config: {type(exc).__name__}: {exc}."
        ) from exc
    labels = annotator(cells)
    return _as_1d_float_tensor(labels, field="recomputed labels", source=source).numpy().astype(np.float64)


def resolve_test_cell_ids(
    payload: dict[str, Any],
    *,
    source: Path,
    config: dict[str, Any] | None,
    y_true: np.ndarray,
    verify_alignment: bool,
    label_tolerance: float = 1e-3,
) -> CellIdResolution:
    """Name the cell behind each prediction, or explain why that is impossible.

    Order of preference: the cells the artifact carries (free, and verifiable);
    then the splitter the config names, which is deterministic and reads only
    filenames (``batteryml/train_test_split/MATR_split.py:15-22``).

    Where the number of cells exceeds the number of predictions, the difference
    is exactly the NaN-label mask of ``batteryml/task.py:62-68``, and the mask is
    reproduced rather than guessed.

    Args:
        payload: the unpickled artifact.
        source: artifact path, for messages.
        config: parsed run config, or ``None``.
        y_true: inverse-transformed labels, used to verify the alignment.
        verify_alignment: load cells from disk when needed to prove the mapping.
            Ignored when the artifact already carries the cells — then the check
            is free and always performed if a config is available.
        label_tolerance: relative tolerance for the label cross-check.

    Returns:
        A :class:`CellIdResolution`. ``cell_ids`` is ``None`` only when identity
        genuinely cannot be established; the reason always says what to pass.

    Raises:
        VizDataError: when the artifact and the config describe different runs.
    """
    n_predictions = int(y_true.shape[0])
    raw_data = payload.get("raw_data")
    cells: list[Any] | None = None
    origin = ""

    if isinstance(raw_data, dict) and isinstance(raw_data.get("test_cells"), (list, tuple)):
        cells = list(raw_data["test_cells"])
        origin = "artifact_raw_data"
    elif config is not None:
        _import_registries()
        from batteryml.builders import TRAIN_TEST_SPLITTERS

        split_config = config.get("train_test_split")
        if not split_config or not dict(split_config).get("name"):
            return CellIdResolution(
                None, "unavailable", None, None, False,
                "the config declares no 'train_test_split.name', so the test cell list "
                "cannot be rebuilt.",
            )
        try:
            splitter = TRAIN_TEST_SPLITTERS.build(dict(split_config), "raise")
            _, test_paths = splitter.split()
        except Exception as exc:
            return CellIdResolution(
                None, "unavailable", None, None, False,
                f"rebuilding the splitter {dict(split_config).get('name')!r} failed: "
                f"{type(exc).__name__}: {exc}. This usually means "
                f"{dict(split_config).get('cell_data_path')} is missing — run preprocessing first.",
            )
        paths = [Path(str(item)) for item in test_paths]
        ids_from_paths = [path.stem for path in paths]
        if len(ids_from_paths) == n_predictions and not verify_alignment:
            return CellIdResolution(
                ids_from_paths, "splitter_paths", len(ids_from_paths), 0, False,
                "cell count equals prediction count, so no cell was dropped by the "
                "NaN-label mask and split order is prediction order. Pass "
                "--verify-cell-alignment to confirm this against the labels.",
            )
        from batteryml.data.battery_data import BatteryData

        try:
            cells = [BatteryData.load(str(path)) for path in paths]
        except Exception as exc:
            return CellIdResolution(
                None, "unavailable", len(paths), None, False,
                f"loading the test cells from {dict(split_config).get('cell_data_path')} "
                f"failed: {type(exc).__name__}: {exc}.",
            )
        origin = "splitter_paths"
    else:
        return CellIdResolution(
            None, "unavailable", None, None, False,
            "the artifact carries no 'raw_data' (the CLI path does not write one, "
            "batteryml/pipeline.py:113-122) and no --config was given, so the test cell "
            "list cannot be rebuilt. Pass --config with the YAML the run used.",
        )

    ids = _cell_ids_of(cells, source=source)
    n_cells = len(ids)
    if n_cells < n_predictions:
        raise VizDataError(
            f"{source}: the test split has {n_cells} cells but the artifact holds "
            f"{n_predictions} predictions. The artifact and the config describe different "
            "runs; pass the config the run was actually launched with."
        )

    if n_cells == n_predictions and config is None:
        return CellIdResolution(
            ids, origin, n_cells, 0, False,
            "cell count equals prediction count, so no cell was dropped by the NaN-label "
            "mask. Pass --config to also verify the mapping against the labels.",
        )

    if config is None:
        return CellIdResolution(
            None, "unavailable", n_cells, None, False,
            f"the test split has {n_cells} cells but only {n_predictions} predictions: "
            "Task.build dropped the difference through its NaN-label mask "
            "(batteryml/task.py:62-68). Pass --config so the label annotator can be "
            "rebuilt and the mask reproduced.",
        )

    labels = _labels_for_cells(cells, config, source=source)
    if labels.shape[0] != n_cells:
        raise VizDataError(
            f"{source}: the label annotator returned {labels.shape[0]} labels for "
            f"{n_cells} cells. This is a defect in the annotator named by the config."
        )
    mask = ~np.isnan(labels)
    kept = [cell_id for cell_id, keep in zip(ids, mask) if keep]
    n_dropped = n_cells - len(kept)
    if len(kept) != n_predictions:
        raise VizDataError(
            f"{source}: reproducing the NaN-label mask keeps {len(kept)} of {n_cells} cells, "
            f"but the artifact holds {n_predictions} predictions. Predictions cannot be "
            "matched to cells, so no per-cell figure was drawn. Confirm the artifact and "
            "the config belong to the same run."
        )

    kept_labels = labels[mask]
    if not np.allclose(kept_labels, y_true, rtol=label_tolerance, atol=1e-6, equal_nan=False):
        worst = int(np.argmax(np.abs(kept_labels - y_true)))
        raise VizDataError(
            f"{source}: the labels recomputed from the test cells do not match the "
            f"artifact's own labels — largest disagreement at position {worst}: "
            f"cell {kept[worst]!r} annotates to {kept_labels[worst]!r} but the artifact "
            f"holds {y_true[worst]!r}. The per-cell mapping would be wrong, so no figure "
            "was drawn. Confirm the artifact, the config and data/processed/ describe the "
            "same run."
        )

    return CellIdResolution(
        kept,
        f"{origin}+label_mask" if n_dropped else origin,
        n_cells,
        n_dropped,
        True,
        f"mask reproduced from the config's label annotator; {n_dropped} of {n_cells} cells "
        "dropped for a NaN label, and every remaining label matches the artifact.",
    )


def load_prediction_artifact(
    artifact_path: Path,
    *,
    config_path: Path | None = None,
    score_tolerance: float = 1e-4,
    verify_cell_alignment: bool = False,
) -> PredictionArtifact:
    """Read one prediction artifact into the label space its metrics live in.

    Args:
        artifact_path: ``predictions_seed_*.pkl`` or
            ``manual_predictions_seed_*.pkl``.
        config_path: the run's YAML config. Optional, but without it cell
            identity and the NaN-label mask cannot always be established.
        score_tolerance: maximum relative difference between the recorded
            metrics and the ones recomputed from the loaded arrays.
        verify_cell_alignment: load cells from disk to prove the per-cell
            mapping even when a count match already implies it.

    Returns:
        A fully validated :class:`PredictionArtifact`.

    Raises:
        VizDataError: for any missing, malformed or inconsistent input. The
            message always names the file and the corrective action.
    """
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise VizDataError(
            f"prediction artifact {artifact_path} does not exist. Run training and "
            "evaluation first; BatteryML writes predictions_seed_<seed>_<ts>.pkl into "
            "the workspace directory."
        )

    try:
        with artifact_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise VizDataError(
            f"cannot unpickle {artifact_path}: {type(exc).__name__}: {exc}. The artifact "
            "contains torch tensors and BatteryML objects, so it must be read with the "
            "BatteryML dependencies installed and the repository root on sys.path."
        ) from exc

    if not isinstance(payload, dict):
        raise VizDataError(
            f"{artifact_path}: expected a dict, found {type(payload).__name__}. This is "
            "not a BatteryML prediction artifact."
        )

    prediction_key = next((key for key in PREDICTION_KEYS if key in payload), None)
    if prediction_key is None:
        raise VizDataError(
            f"{artifact_path}: no prediction found under any of {PREDICTION_KEYS}. "
            f"Keys present: {sorted(payload)}. Point --prediction at a file written by "
            "Pipeline.evaluate or by scripts/reproduce_batteryml/manual_train_eval.py."
        )

    bundle = payload.get("data")
    if bundle is None or not hasattr(bundle, "test_data"):
        raise VizDataError(
            f"{artifact_path}: no DataBundle under 'data' (found "
            f"{type(payload.get('data')).__name__}). The test labels live there, so the "
            "artifact cannot be plotted. Re-run evaluation with a workspace configured."
        )

    y_pred_t = _as_1d_float_tensor(payload[prediction_key], field=prediction_key, source=artifact_path)
    y_true_t = _as_1d_float_tensor(bundle.test_data.label, field="data.test_data.label", source=artifact_path)
    if y_pred_t.numel() != y_true_t.numel():
        raise VizDataError(
            f"{artifact_path}: {y_pred_t.numel()} predictions but {y_true_t.numel()} test "
            f"labels. The artifact is internally inconsistent; re-run the evaluation."
        )

    transformation = getattr(bundle, "label_transformation", None)
    chain = _transformation_chain(transformation)
    chain_text = " -> ".join(chain) if chain else "the declared label transformation"
    if transformation is not None:
        try:
            y_true_t = _as_1d_float_tensor(
                transformation.inverse_transform(y_true_t),
                field="inverse-transformed labels",
                source=artifact_path,
            )
            y_pred_t = _as_1d_float_tensor(
                transformation.inverse_transform(y_pred_t),
                field="inverse-transformed predictions",
                source=artifact_path,
            )
        except VizDataError:
            raise
        except Exception as exc:
            raise VizDataError(
                f"{artifact_path}: inverse-transforming with {chain_text} failed: "
                f"{type(exc).__name__}: {exc}."
            ) from exc

    non_finite = int((~torch.isfinite(y_true_t)).sum() + (~torch.isfinite(y_pred_t)).sum())
    if non_finite:
        raise VizDataError(
            f"{artifact_path}: {non_finite} non-finite value(s) after inverse-transforming "
            f"with {chain_text if chain else 'no transformation'}. An exp() overflow in a "
            "LogScale inverse is the usual cause. No figure was drawn."
        )

    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict) or not raw_scores:
        raise VizDataError(
            f"{artifact_path}: 'scores' is missing or empty (found "
            f"{type(payload.get('scores')).__name__}). Figures display BatteryML's own "
            "metrics and never recompute them, so the artifact is unusable. Re-run "
            "evaluation with --eval."
        )
    scores = {str(key): float(value) for key, value in raw_scores.items()}
    score_check = verify_recorded_scores(
        y_true_t, y_pred_t, scores, tolerance=score_tolerance, source=artifact_path
    )

    y_true = y_true_t.numpy().astype(np.float64)
    y_pred = y_pred_t.numpy().astype(np.float64)

    config = read_config(config_path) if config_path is not None else None
    sample_index = _sample_index_of(payload, artifact_path, len(y_true))
    if sample_index is not None:
        # The run recorded who each row belongs to, so there is nothing to
        # infer. The whole-cell resolver must not run here at all: it compares
        # the test split's *cell* count against the row count, which for a
        # cycle-level task is 5 cells against 55 rows — correct, and rejected
        # as an inconsistency. The recorded identity is authoritative.
        cells = CellIdResolution(
            cell_ids=[record["cell_id"] for record in sample_index],
            source="artifact_sample_index",
            n_cells_total=len({record["cell_id"] for record in sample_index}),
            n_dropped_nan_label=None,
            alignment_verified=True,
            reason=(
                "the artifact records a cell id and observed cycle for every "
                "prediction, written by the evaluation step; nothing is inferred."
            ),
        )
    else:
        cells = resolve_test_cell_ids(
            payload,
            source=artifact_path,
            config=config,
            y_true=y_true,
            verify_alignment=verify_cell_alignment,
        )

    seed = payload.get("seed")
    return PredictionArtifact(
        source_path=artifact_path,
        source_sha256=sha256_of(artifact_path),
        seed=int(seed) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else None,
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
        score_check=score_check,
        label_transformations=chain,
        cells=cells,
        sample_index=sample_index,
        task_metadata=dict(payload.get('task_metadata') or {})
        if isinstance(payload.get('task_metadata'), dict) else {},
    )
