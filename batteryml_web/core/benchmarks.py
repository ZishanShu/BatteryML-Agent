"""Published BatteryML RUL benchmark values, and when a run may be compared.

**Provenance.** The table below is transcribed from this repository's own
``README.md``, section *"Benchmark result of RUL(Remain Useful Life) task"*.
It is not reproduced from anyone's memory of the paper, and
``tests/test_benchmark_registry.py`` re-parses that README on every run and
fails if a single cell here drifts from it.

**What was verified, and how.** The README states the task ("RUL") and the
seed protocol ("the error mean across ten seeds ... standard deviation as
subscript") but never names the error metric. ``result.ipynb`` — the
notebook in this repository that *generated* the table — does:

* the metric is RMSE. Every cell comes from
  ``extract_scores(...)['RMSE']``;
* sklearn models use one seed (``log.0``), formatted ``int(score)``, which
  is why they carry no dispersion;
* NN models use ten seeds (``range(10)``), formatted ``f'{mean:.0f}±{std:.0f}'``;
* its dataset keys are exactly ``['clo','cruh','crush','snl','matr_1',
  'matr_2','hust','mix']`` — the config basenames — which is what ties the
  ``MATR1``/``MATR2`` columns to ``configs/.../matr_1.yaml`` and
  ``matr_2.yaml`` rather than to a guess.

``baseline.ipynb`` corroborates the subset mapping independently: its dummy
regressor scores ``test_RMSE 398.82`` and ``test2_RMSE 510.63``, matching the
README's MATR1 ``398`` and MATR2 ``510`` for that row.

The unit is cycles: RUL labels are cycle counts (``RULLabelAnnotator``), and
``docs/VISUALIZATION.md`` records that the stored RMSE is a number of cycles.

**What could not be verified here.** The paper itself is unreachable from
this environment (arxiv.org and openreview.net are both blocked by the
egress proxy) and no PDF is vendored. So the exact wording of ``>1000`` is
not sourced: it is evidently a censoring threshold for runs whose error
diverged, but whether it means strictly greater than 1000, or a run that
failed to converge, is not stated anywhere local. It is therefore stored as
a censored value with **no central value at all** — never as the number
1000 — and a censored reference can never produce a numeric difference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

#: Where every value here comes from. Quoted in the UI beside the number.
BENCHMARK_SOURCE = (
    "BatteryML README.md, “Benchmark result of RUL(Remain Useful Life) "
    "task” (Zhang et al., ICLR 2024)"
)

#: The one task this registry describes. A run on any other task has no
#: reference here — which is the whole of the SOH rule: SOH error is a ratio
#: of capacities and these are cycle counts.
BENCHMARK_TASK = "RUL prediction"

BENCHMARK_METRIC = "RMSE"
BENCHMARK_UNIT = "cycles"

#: Values above this were reported as ">1000" rather than as a number.
CENSOR_THRESHOLD = 1000.0

#: Seeds behind a value, decided by the README's own stated rule: "For
#: models sensitive to initialization, we present the error mean across ten
#: seeds and attach the standard deviation as subscript." So a cell that
#: carries a dispersion is a ten-seed mean, and a cell without one is a
#: single run.
#:
#: This is derived per cell from the printed value rather than from
#: ``result.ipynb``'s model grouping, because the two disagree and the
#: README is the published artifact. The notebook lists ``rf`` among the
#: single-seed sklearn baselines and formats that group as ``int(score)``
#: with no dispersion — yet the README's Random forest row reads ``168±9``.
#: The saved notebook therefore does not reproduce that row, so its grouping
#: cannot be used to say how many seeds are behind a value; the README's
#: sentence can, and it agrees with every cell.
SEEDS_SKLEARN = 1
SEEDS_MULTI = 10


@dataclass(frozen=True)
class BenchmarkValue:
    """One published cell.

    ``central`` is ``None`` exactly when ``censored`` is True: ">1000" is not
    a measurement, and turning it into ``1000.0`` would invent one. Callers
    that want to show something still have ``censor_threshold``.
    """

    central: Optional[float]
    dispersion: Optional[float]
    #: None for a censored cell: nothing is known about what produced it.
    seeds: Optional[int]
    censored: bool = False
    censor_threshold: Optional[float] = None

    def display(self) -> str:
        """How the value is written, in the README's own notation."""
        if self.censored:
            return f">{self.censor_threshold:.0f}"
        if self.central is None:
            return "unavailable"
        if self.dispersion is None:
            return f"{self.central:.0f}"
        return f"{self.central:.0f} ± {self.dispersion:.0f}"


@dataclass(frozen=True)
class BenchmarkRecord:
    """A published reference for one (model, dataset subset) pair."""

    source: str
    task: str
    dataset_subset: str
    model_key: str
    metric: str
    unit: str
    lower_is_better: bool
    value: BenchmarkValue
    protocol: str


#: The benchmark columns, and the train/test splitter class that defines
#: each one. **This mapping is the answer to "MATR1 or MATR2?"** — it is read
#: from the run's own split identity, never from the dataset name and never
#: from the model. ``MATRTrainTestSplitter`` (the generic parent) is
#: deliberately absent: a run using it is on a MATR split that is not one of
#: the two published subsets, so it has no column.
SUBSET_BY_SPLITTER: Dict[str, str] = {
    "MATRPrimaryTestTrainTestSplitter": "MATR1",
    "MATRSecondaryTestTrainTestSplitter": "MATR2",
    "MATRCLOTestTrainTestSplitter": "CLO",
    "HUSTTrainTestSplitter": "HUST",
    "SNLTrainTestSplitter": "SNL",
    "CRUHTrainTestSplitter": "CRUH",
    "CRUSHTrainTestSplitter": "CRUSH",
    "MIX100TrainTestSplitter": "MIX",
}

#: Column order, as the README prints it.
SUBSETS: Tuple[str, ...] = (
    "MATR1", "MATR2", "HUST", "SNL", "CLO", "CRUH", "CRUSH", "MIX",
)

#: The README's row label for each platform model key. Explicit, and keyed
#: on the platform's own ``model_key`` — never on string similarity.
#:
#: What is *absent* matters as much as what is present. ``cptransformer``,
#: ``batlinet``, ``PBT`` and ``BatteryMFormer`` have no row: CPTransformer is
#: not the paper's "Transformer" and BatLiNet is not its "MLP", however
#: similar the names read. A model missing from this mapping is reported as
#: having no published counterpart, which is true, rather than compared with
#: an unrelated one.
ROW_BY_MODEL_KEY: Dict[str, str] = {
    "dummy": "Dummy regressor",
    "variance_model": '"Variance" model',
    "discharge_model": '"Discharge" model',
    "full_model": '"Full" model',
    "ridge": "Ridge regression",
    "pcr": "PCR",
    "plsr": "PLSR",
    "gpr": "Gaussian process",
    "xgb": "XGBoost",
    "rf": "Random forest",
    "mlp": "MLP",
    "cnn": "CNN",
    "lstm": "LSTM",
    "transformer": "Transformer",
}

#: The table, transcribed from README.md in its own notation. Parsed below.
#: Kept as the printed strings rather than as pre-split numbers so that the
#: drift test can compare it against the README without re-deriving anything.
_TABLE: Dict[str, Tuple[str, ...]] = {
    "Dummy regressor":    ("398", "510", "419", "466", "331", "239", "576", "573"),
    '"Variance" model':   ("136", "211", "398", "360", "179", "118", "506", "521"),
    '"Discharge" model':  ("329", "149", "322", "267", "143", "76", ">1000", ">1000"),
    '"Full" model':       ("167", ">1000", "335", "433", "138", "93", ">1000", "331"),
    "Ridge regression":   ("116", "184", ">1000", "242", "169", "65", ">1000", "372"),
    "PCR":                ("90", "187", "435", "200", "197", "68", "560", "376"),
    "PLSR":               ("104", "181", "431", "242", "176", "60", "535", "383"),
    "Gaussian process":   ("154", "224", ">1000", "251", "204", "115", ">1000", "573"),
    "XGBoost":            ("334", "799", "395", "547", "215", "119", "330", "205"),
    "Random forest":      ("168±9", "233±7", "368±7", "532±25", "192±2", "81±1", "416±5", "197±0"),
    "MLP":                ("149±3", "275±27", "459±9", "370±81", "146±5", "103±4", "565±9", "451±42"),
    "CNN":                ("102±94", "228±104", "465±75", "924±267", ">1000", "174±92", "545±11", "272±101"),
    "LSTM":               ("119±11", "219±33", "443±29", "539±40", "222±12", "105±10", "519±39", "268±9"),
    "Transformer":        ("135±13", "364±25", "391±11", "424±23", "187±14", "81±8", "550±21", "271±16"),
}

_CELL = re.compile(r"^(?P<central>\d+(?:\.\d+)?)(?:±(?P<std>\d+(?:\.\d+)?))?$")


def parse_cell(cell: str) -> BenchmarkValue:
    """Turn one printed table cell into a value.

    The seed count is read from the notation, per the README's rule: a
    dispersion means ten seeds, its absence means one.

    ``">1000"`` becomes a censored value with no central number. That is the
    point: a censored cell must not be silently readable as 1000, because a
    run scoring 900 would then look like an improvement over a run the paper
    declined to score at all. A censored cell records no seed count either —
    nothing is known about what produced it.
    """
    text = str(cell).strip()
    if text.startswith(">"):
        return BenchmarkValue(
            central=None,
            dispersion=None,
            seeds=None,
            censored=True,
            censor_threshold=float(text[1:]),
        )
    match = _CELL.match(text)
    if match is None:
        raise ValueError(f"Unparseable benchmark cell: {cell!r}")
    std = match.group("std")
    return BenchmarkValue(
        central=float(match.group("central")),
        dispersion=float(std) if std is not None else None,
        seeds=SEEDS_MULTI if std is not None else SEEDS_SKLEARN,
    )


def _build() -> Dict[Tuple[str, str], BenchmarkRecord]:
    records: Dict[Tuple[str, str], BenchmarkRecord] = {}
    protocol_by_subset = {
        subset: splitter for splitter, subset in SUBSET_BY_SPLITTER.items()
    }
    for model_key, row in ROW_BY_MODEL_KEY.items():
        for subset, cell in zip(SUBSETS, _TABLE[row]):
            records[(model_key, subset)] = BenchmarkRecord(
                source=BENCHMARK_SOURCE,
                task=BENCHMARK_TASK,
                dataset_subset=subset,
                model_key=model_key,
                metric=BENCHMARK_METRIC,
                unit=BENCHMARK_UNIT,
                lower_is_better=True,
                value=parse_cell(cell),
                protocol=protocol_by_subset[subset],
            )
    return records


_RECORDS = _build()


def subset_for_splitter(splitter_name: str | None) -> Optional[str]:
    """The benchmark column a split class corresponds to, or None.

    None is a real answer, not a failure: a MATR run on the generic splitter
    is on neither published subset, and saying so is what stops MATR1 being
    picked because it is listed first.
    """
    return SUBSET_BY_SPLITTER.get(str(splitter_name or "").strip())


def lookup(model_key: str | None, subset: str | None) -> Optional[BenchmarkRecord]:
    """The published record for this exact model key and subset, or None."""
    if not model_key or not subset:
        return None
    return _RECORDS.get((str(model_key).strip(), str(subset).strip()))


def all_records() -> Tuple[BenchmarkRecord, ...]:
    """Every record, for the reference table and for tests."""
    return tuple(_RECORDS.values())


# --- comparability -------------------------------------------------------

#: Why a comparison was refused. One code per reason, so the UI and the
#: prompt say the same thing and a test can assert on the cause rather than
#: on prose.
REASON_TASK = "task_not_benchmarked"
REASON_SUBSET = "subset_not_identified"
REASON_MODEL = "model_not_published"
REASON_METRIC = "metric_differs"
REASON_SMOKE = "smoke_run"
REASON_CENSORED = "reference_censored"
REASON_NO_VALUE = "run_metric_unavailable"

#: Human wording for each code. Kept beside the codes so there is one
#: sentence per reason in the whole application.
REASON_TEXT: Dict[str, str] = {
    REASON_TASK: (
        "the published BatteryML benchmark covers RUL prediction only, and "
        "this run is a different task"
    ),
    REASON_SUBSET: (
        "the run's train/test split does not identify one of the published "
        "benchmark subsets"
    ),
    REASON_MODEL: "this model has no directly corresponding published BatteryML baseline",
    REASON_METRIC: "the run's metric or unit is not the benchmark's RMSE in cycles",
    REASON_SMOKE: (
        "this is a 1-epoch smoke run, which validates the workflow rather "
        "than measuring trained performance"
    ),
    REASON_CENSORED: (
        "the published value for this cell is reported as a threshold "
        "(“>1000”) rather than as a number"
    ),
    REASON_NO_VALUE: "this run has no saved RMSE to compare",
}


@dataclass(frozen=True)
class Comparability:
    """Whether a published reference may be set beside this run's number.

    ``comparable`` gates the *arithmetic*. ``record`` may be present while
    ``comparable`` is False — a reference can be worth showing as context
    even when subtracting it would be wrong, which is the case a smoke run
    creates. ``difference`` is None unless ``comparable``.
    """

    comparable: bool
    record: Optional[BenchmarkRecord]
    reasons: Tuple[str, ...]
    run_value: Optional[float] = None
    difference: Optional[float] = None

    def explain(self) -> str:
        """One sentence naming every reason, or the positive statement."""
        if self.comparable:
            return (
                "Directly comparable: same task, benchmark subset, metric "
                "and unit, and a fully trained run."
            )
        if not self.reasons:
            return "Direct comparison is unavailable."
        return "Direct comparison is unavailable because " + "; ".join(
            REASON_TEXT.get(code, code) for code in self.reasons
        ) + "."


def assess_comparability(
    *,
    task: str | None,
    splitter_name: str | None,
    model_key: str | None,
    metric_name: str | None,
    metric_value: Optional[float] = None,
    unit: str | None = None,
    is_smoke: bool = False,
) -> Comparability:
    """Decide whether this run may be compared with a published value.

    Every condition is checked and *all* failures are reported, rather than
    returning on the first: "not comparable because the split differs" is
    less useful than knowing the metric differs too.

    The rule this exists to enforce: an SOH run is never measured against a
    RUL benchmark. SOH error is a ratio of capacities of order 0.1 and these
    references are cycle counts of order 100, so printing one beside the
    other would invite exactly the comparison that is meaningless. That case
    fails on ``task`` before anything else is even looked up.
    """
    reasons: list[str] = []

    task_matches = str(task or "").strip() == BENCHMARK_TASK
    if not task_matches:
        reasons.append(REASON_TASK)

    subset = subset_for_splitter(splitter_name)
    if subset is None:
        reasons.append(REASON_SUBSET)

    # **The lookup is gated on the task.** A ridge/MATR1 SOH run would
    # otherwise resolve the ridge/MATR1 *RUL* record — same model, same
    # split — and the caller would then put "116 cycles" on screen beside
    # an SOH RMSE of 0.081. Labelling that pair "not comparable" is not
    # enough: the two numbers differ by three orders of magnitude and
    # showing them together is the error, whatever the caption says. A
    # different task gets no record at all.
    record = lookup(model_key, subset) if (task_matches and subset) else None
    if record is None and task_matches:
        reasons.append(REASON_MODEL)

    if str(metric_name or "").strip().upper() != BENCHMARK_METRIC or (
        unit is not None and str(unit).strip() != BENCHMARK_UNIT
    ):
        reasons.append(REASON_METRIC)

    if is_smoke:
        reasons.append(REASON_SMOKE)

    if record is not None and record.value.censored:
        reasons.append(REASON_CENSORED)

    if metric_value is None:
        reasons.append(REASON_NO_VALUE)

    if reasons:
        # The record still travels when one was found: a reference worth
        # naming as context is not the same as a reference worth subtracting.
        return Comparability(
            comparable=False,
            record=record,
            reasons=tuple(reasons),
            run_value=metric_value,
        )

    assert record is not None and record.value.central is not None
    return Comparability(
        comparable=True,
        record=record,
        reasons=(),
        run_value=metric_value,
        difference=float(metric_value) - record.value.central,
    )
