# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""Who each row of a feature/label tensor belongs to.

BatteryML's original contract is one sample per cell: ``feature[i]`` is cell
``i``, so the row index *is* the identity and nothing needs to be recorded.
A cycle-level task breaks that — one cell contributes many rows — and the
moment it does, a prediction can no longer be traced back to the cell and
cycle it was made for.

:class:`SampleIndex` is that record: a parallel array of ``(cell_id,
cycle_index)`` whose length is the leading dimension of the feature and label
tensors. It is deliberately not a tensor. ``cell_id`` is a string, it is never
fed to a model, and keeping it out of the numeric path means it cannot be
mistaken for a feature.

Identity is optional everywhere it appears: an existing one-sample-per-cell
RUL run carries ``None`` and behaves exactly as it did before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

#: The cycle index recorded for a sample that is not cycle-level — a
#: whole-cell label such as RUL, which is about the cell, not about a cycle.
WHOLE_CELL_CYCLE = -1


class SampleIndexError(ValueError):
    """A sample index does not match the tensors it is supposed to describe.

    Always names the two lengths, because the only way this happens is a
    feature extractor and a label annotator disagreeing about how many
    samples a cell produced.
    """


@dataclass(frozen=True)
class SampleIndex:
    """Identity for each row of a feature/label tensor, in row order.

    Attributes:
        cell_ids: the cell each row came from. Order is the tensor's order.
        cycle_indices: the observed cycle each row was taken at, using
            BatteryML's own 1-based ``CycleData.cycle_number`` convention, or
            :data:`WHOLE_CELL_CYCLE` for a whole-cell sample.
    """

    cell_ids: tuple[str, ...]
    cycle_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.cell_ids) != len(self.cycle_indices):
            raise SampleIndexError(
                f"SampleIndex has {len(self.cell_ids)} cell id(s) but "
                f"{len(self.cycle_indices)} cycle index(es). Every sample "
                "needs exactly one of each."
            )

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, item: int) -> dict:
        return {
            "cell_id": self.cell_ids[item],
            "cycle_index": self.cycle_indices[item],
        }

    @property
    def unique_cell_ids(self) -> List[str]:
        """The distinct cells present, in first-appearance order."""
        seen: dict[str, None] = {}
        for cell_id in self.cell_ids:
            seen.setdefault(cell_id, None)
        return list(seen)

    @classmethod
    def for_cell(cls, cell_id: str, cycle_indices: Sequence[int]) -> "SampleIndex":
        """The index for one cell's cycle-level samples."""
        cycles = tuple(int(cycle) for cycle in cycle_indices)
        return cls(cell_ids=tuple([str(cell_id)] * len(cycles)), cycle_indices=cycles)

    @classmethod
    def for_whole_cells(cls, cell_ids: Sequence[str]) -> "SampleIndex":
        """The index for the classic one-sample-per-cell layout."""
        ids = tuple(str(cell_id) for cell_id in cell_ids)
        return cls(cell_ids=ids, cycle_indices=tuple([WHOLE_CELL_CYCLE] * len(ids)))

    @classmethod
    def concat(cls, parts: Iterable["SampleIndex"]) -> "SampleIndex":
        """Join per-cell indices in order, matching a ``torch.cat`` of rows."""
        cell_ids: list[str] = []
        cycles: list[int] = []
        for part in parts:
            cell_ids.extend(part.cell_ids)
            cycles.extend(part.cycle_indices)
        return cls(cell_ids=tuple(cell_ids), cycle_indices=tuple(cycles))

    def select(self, mask: Sequence[bool]) -> "SampleIndex":
        """The index for the rows ``mask`` keeps, e.g. after NaN filtering."""
        if len(mask) != len(self):
            raise SampleIndexError(
                f"Mask of length {len(mask)} cannot select from a SampleIndex "
                f"of length {len(self)}."
            )
        kept = [i for i, keep in enumerate(mask) if keep]
        return SampleIndex(
            cell_ids=tuple(self.cell_ids[i] for i in kept),
            cycle_indices=tuple(self.cycle_indices[i] for i in kept),
        )

    def check_matches(self, n_rows: int, what: str) -> None:
        """Raise unless this index describes exactly ``n_rows`` rows."""
        if len(self) != n_rows:
            raise SampleIndexError(
                f"{what}: {n_rows} row(s) but {len(self)} sample identity "
                "record(s). A feature extractor and a label annotator "
                "disagreed about how many samples a cell produces; they must "
                "be driven by the same cycle list."
            )

    def to_records(self) -> List[dict]:
        """A plain, picklable list of dicts, for a result artifact."""
        return [self[i] for i in range(len(self))]


def assert_disjoint_cells(splits: dict[str, Sequence[str]]) -> None:
    """Raise unless no cell appears in more than one split.

    The scientific hard rule for a cycle-level task: cells are split *before*
    they are expanded into cycles. If one cell's cycles land on both sides of
    the split, the test score measures memorisation of that cell rather than
    generalisation to a new one, and nothing about the run will look wrong.

    Args:
        splits: split name -> the cell ids assigned to it.

    Raises:
        SampleIndexError: naming every offending cell and the splits it is in.
    """
    names = list(splits)
    offenders: dict[str, list[str]] = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = set(splits[left]) & set(splits[right])
            for cell_id in sorted(shared):
                offenders.setdefault(cell_id, [])
                for name in (left, right):
                    if name not in offenders[cell_id]:
                        offenders[cell_id].append(name)
    if offenders:
        detail = "; ".join(
            f"{cell_id} in {'+'.join(where)}" for cell_id, where in sorted(offenders.items())
        )
        raise SampleIndexError(
            f"{len(offenders)} cell(s) appear in more than one split: {detail}. "
            "Cells must be split before they are expanded into cycle-level "
            "samples; splitting expanded rows leaks a cell across the split "
            "and inflates the score."
        )
