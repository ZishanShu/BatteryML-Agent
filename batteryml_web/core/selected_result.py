"""One identity for "the result the user is looking at".

The defect this exists for, reproduced before it was fixed: Review Results
drew its Metrics, Summary and Logs tabs — and built both LLM prompts — from
``find_result_files(project_dir, …)``, which ``rglob``\\s the **whole
project** and hands back every run's artifacts sorted by path. The readers
above it then took ``[0]``. Run ids begin with a timestamp, so path order is
chronological and ``[0]`` is the *oldest run in the project*. Selecting a run
in Run History changed none of it.

Three of the reported symptoms fall out of that one mechanism:

* selecting run B leaves the metrics showing run A — they were never B's;
* predictions "look identical between runs" — they are literally the same
  file, re-read every time;
* the explanation describes another experiment — its prompt was built from
  that same oldest-run table.

Meanwhile the Predictions and Files tabs *did* read the selected run's
manifest, so one page showed two different runs at once under one heading.

The repair is this module: a single value that names exactly one result, and
which every section of the page derives from. It is deliberately built from
the **run's own manifest**, not from a directory search:

* the run id and run directory come from the manifest the history row was
  read from;
* the prediction artifact comes from that manifest's recorded
  ``artifacts.prediction_files``;
* where a legacy run recorded none, discovery is scoped **inside that run's
  directory** and must find exactly one candidate, or the caller must name
  one. There is no "newest file in the project" fallback, and no "the last
  artifact that worked" fallback: if the selected run has no readable
  predictions, that is what the page says.

``identity_key`` includes the artifact's size and mtime, so a rewritten
artifact is a different identity and cannot be served from a cache keyed on
the previous one. It includes the run id, so two runs of an identical
configuration and seed — same dataset, same model, same everything but their
identity — never collide.

No Streamlit, no pickle, no LLM: this decides *which* file is the subject and
says why when there is none. Reading it stays with the CPU-safe loader in
:mod:`batteryml_web.agent.results`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Filenames a legacy run directory may hold in place of a recorded artifact
#: reference. Matched **only inside the selected run's own directory**.
LEGACY_PREDICTION_PATTERNS: tuple[str, ...] = (
    "predictions_seed_*.pkl",
    "manual_predictions_seed_*.pkl",
    "*predictions*.pkl",
    "*prediction*.pkl",
    "*predictions*.csv",
    "*prediction*.csv",
)

#: What the page says when the selected run cannot supply a result. Each is a
#: statement about *this* run; none of them is a reason to show another run.
NO_RUN_SELECTED = "No run is selected. Choose one in Run History above."
NO_ARTIFACT = (
    "This run recorded no predictions artifact, so there is no result to "
    "show for it."
)
ARTIFACT_MISSING = "The prediction artifact this run recorded is not on disk."
AMBIGUOUS_ARTIFACT = (
    "This run offers more than one prediction artifact; choose which one to "
    "read."
)
OUTSIDE_RUN_DIR = (
    "The prediction artifact this run recorded lies outside its own run "
    "directory and was not read."
)


@dataclass(frozen=True)
class SelectedResult:
    """The one result every section of the page must describe.

    ``available`` is False whenever this run cannot supply a result, and
    ``reason`` then says why **about this run**. A caller that sees
    ``available is False`` shows that reason; it never reaches for another
    run's artifact, which is the substitution that made the page lie.
    """

    project_name: str = ""
    project_root: str = ""
    run_id: str = ""
    run_dir: str = ""
    predictions_file: Optional[str] = None
    #: Size and modification time of the artifact, so a rewritten file is a
    #: different identity rather than a cache hit on the previous one.
    artifact_revision: str = ""
    #: Every prediction artifact this run offers, for an explicit choice.
    candidates: tuple[str, ...] = ()
    manifest: Mapping[str, Any] = field(default_factory=dict)
    reason: str = NO_RUN_SELECTED

    @property
    def available(self) -> bool:
        """Whether a readable prediction artifact was resolved for this run."""
        return bool(self.predictions_file) and not self.reason

    def identity_key(self) -> str:
        """A stable key naming exactly this result.

        Carries the project, the run id, the artifact path and the artifact
        revision. The run id is what keeps two runs of an identical
        configuration and seed apart; the revision is what keeps a rewritten
        artifact from being served from the previous one's cache entry.

        Deliberately **not** derived from dataset/model/config/seed. Those
        describe an experiment, and a plan signature built from them cannot
        distinguish two executions of it — which is exactly the confusion
        this module exists to end.
        """
        return "|".join(
            [
                self.project_root or self.project_name,
                self.run_id,
                self.predictions_file or "",
                self.artifact_revision,
            ]
        )

    def provenance_rows(self) -> List[Dict[str, str]]:
        """The compact identification block the result panel shows."""
        manifest = self.manifest or {}
        return [
            {"Field": "Project", "Value": self.project_name or "Not available"},
            {"Field": "Run ID", "Value": self.run_id or "Not available"},
            {
                "Field": "Model",
                "Value": str(
                    manifest.get("model_display")
                    or manifest.get("model_key")
                    or "Not available"
                ),
            },
            {"Field": "Dataset", "Value": str(manifest.get("dataset") or "Not available")},
            {"Field": "Task", "Value": str(manifest.get("task") or "Not available")},
            {"Field": "Run directory", "Value": self.run_dir or "Not available"},
        ]


def _revision_of(path: Path) -> str:
    """``<size>:<mtime_ns>`` for ``path``, or "" when it cannot be read.

    Cheap, and enough: a re-run that overwrites an artifact in place changes
    at least one of the two, so the identity changes with it.
    """
    try:
        stat = path.stat()
    except OSError:
        return ""
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _within(path: Path, root: Path) -> bool:
    """Whether ``path`` is inside ``root``, resolved.

    The containment check that stops a manifest — hand-edited, or written by
    an older version — from pointing the page at another run's artifact.
    """
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def recorded_artifacts(manifest: Mapping[str, Any]) -> List[str]:
    """The prediction artifacts this run's manifest names, in recorded order."""
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return []
    files = artifacts.get("prediction_files")
    if not isinstance(files, (list, tuple)):
        return []
    return [str(entry) for entry in files if str(entry).strip()]


def discover_in_run_dir(run_dir: Path) -> List[str]:
    """Prediction artifacts inside **this run's** directory, sorted.

    The legacy path, for a run recorded before manifests carried artifact
    references. Scoped to the one directory on purpose: a project-wide search
    is what produced the wrong run's numbers in the first place, and widening
    it again to "be helpful" would restore the defect.
    """
    if not run_dir.is_dir():
        return []
    found: List[Path] = []
    seen: set = set()
    for pattern in LEGACY_PREDICTION_PATTERNS:
        for path in sorted(run_dir.rglob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                found.append(path)
    return [str(path) for path in found]


def resolve_selected_result(
    project: Mapping[str, Any],
    manifest: Optional[Mapping[str, Any]],
    *,
    chosen_artifact: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> SelectedResult:
    """The authoritative identity for the run the user opened.

    Args:
        project: the resolved project (``name`` and ``root``), so a selection
            can never be read against a different project's directory.
        manifest: the selected run's manifest, or None when nothing is
            selected.
        chosen_artifact: the artifact the user picked, when the run offers
            more than one. A value that is not one of this run's own
            candidates is refused rather than honoured.
        repo_root: base for resolving a relative artifact path.

    Never returns another run's artifact. Every failure returns an
    unavailable result carrying the reason.
    """
    name = str(project.get("name") or "")
    root = str(project.get("root") or "")
    if not isinstance(manifest, Mapping) or not manifest:
        return SelectedResult(
            project_name=name, project_root=root, reason=NO_RUN_SELECTED
        )

    run_id = str(manifest.get("run_id") or "")
    run_dir_text = str(manifest.get("run_dir") or manifest.get("workspace") or "")
    base = Path(repo_root) if repo_root is not None else Path.cwd()
    run_dir = Path(run_dir_text)
    if run_dir_text and not run_dir.is_absolute():
        run_dir = base / run_dir

    common = {
        "project_name": name,
        "project_root": root,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": manifest,
    }

    candidates = recorded_artifacts(manifest) or discover_in_run_dir(run_dir)
    if not candidates:
        return SelectedResult(**common, reason=NO_ARTIFACT)

    if chosen_artifact is not None and str(chosen_artifact) in candidates:
        target_text = str(chosen_artifact)
    elif len(candidates) == 1:
        target_text = candidates[0]
    else:
        # More than one, and nothing valid chosen. The page asks rather than
        # picking "the first" — which is the habit that produced the bug.
        return SelectedResult(
            **common,
            candidates=tuple(candidates),
            reason=AMBIGUOUS_ARTIFACT,
        )

    target = Path(target_text)
    if not target.is_absolute():
        target = base / target
    if run_dir_text and not _within(target, run_dir):
        return SelectedResult(
            **common, candidates=tuple(candidates), reason=OUTSIDE_RUN_DIR
        )
    if not target.is_file():
        return SelectedResult(
            **common,
            candidates=tuple(candidates),
            reason=f"{ARTIFACT_MISSING} Expected: {target}",
        )

    return SelectedResult(
        **common,
        predictions_file=str(target),
        artifact_revision=_revision_of(target),
        candidates=tuple(candidates),
        reason="",
    )


def belongs_to(identity_key: str, selected: SelectedResult) -> bool:
    """Whether a stored, result-derived value still describes ``selected``.

    Used to invalidate anything cached against an earlier selection — an
    explanation above all. A late arrival for run A, landing after the user
    has moved to run B, fails this check and is discarded from B's panel
    rather than being displayed under B's heading.
    """
    return bool(identity_key) and identity_key == selected.identity_key()


def cached_for(
    store: Mapping[str, Any],
    value_key: str,
    identity_key_field: str,
    selected: SelectedResult,
) -> Optional[Any]:
    """A stored value, but only if it was stored for **this** result.

    ``store`` is the session state. ``value_key`` holds the value and
    ``identity_key_field`` holds the identity it was computed for. A mismatch
    returns None, which is how the page shows nothing rather than showing the
    previous run's answer under the current run's title.
    """
    recorded = str(store.get(identity_key_field) or "")
    if not belongs_to(recorded, selected):
        return None
    return store.get(value_key)


def artifact_choices(selected: SelectedResult) -> Sequence[str]:
    """The artifacts the user may choose between for this run."""
    return selected.candidates
