"""Shared pieces for the standalone Fig. 3 panel scripts.

Nothing here reads a prediction artifact: that is ``collect_fig3_data.py``,
through the platform's existing CPU-safe loader. The panel scripts read only
the standardized CSV/JSON files ``collect`` writes, so every number a panel
draws can be traced to one row of ``runs.csv`` / ``samples.csv`` and, from
there, to one exact artifact.

Style: white background, Matplotlib's bundled DejaVu Sans, editable SVG text
(``svg.fonttype = "none"``), deterministic output through
``batteryml_web.viz.render.render_to_bytes``. Colours: the validated
8-slot categorical palette, assigned to models by their position in the
selection's ``model_order`` — never by rank, and never cycled.
"""

from __future__ import annotations

import csv
import json
import math
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Panel statuses reported by plot_fig3_all.py.
CREATED = "CREATED"
MISSING_DATA = "MISSING_DATA"
INVALID_COMPARISON = "INVALID_COMPARISON"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

#: Validated categorical order (light surface): blue, orange, aqua, yellow,
#: magenta, green, violet, red. Slot = position in model_order.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948")
#: Secondary encoding, so identity never rests on colour alone.
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
#: Single-hue sequential ramp (blue 100 -> 700) for magnitude.
SEQUENTIAL = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NA_FILL = "#f0efec"
CENSORED_FILL = "#d6d4cd"
SURFACE = "#ffffff"

SMOKE_NOTE = "workflow-validation example (includes smoke runs; not benchmark performance)"


class PanelSkip(Exception):
    """A panel that must not be drawn; ``status`` says which kind of no."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class Style:
    """Figure size, font size and line width; all configurable from the CLI."""

    width_in: float = 3.5
    height_in: float = 2.8
    font_size: float = 7.0
    line_width: float = 0.8
    dpi: float = 300.0

    def rc(self) -> Dict[str, Any]:
        return {
            "svg.fonttype": "none",          # editable text in Illustrator
            "font.family": "DejaVu Sans",
            "font.size": self.font_size,
            "axes.titlesize": self.font_size + 1,
            "axes.labelsize": self.font_size,
            "xtick.labelsize": self.font_size - 0.5,
            "ytick.labelsize": self.font_size - 0.5,
            "legend.fontsize": self.font_size - 0.5,
            "axes.linewidth": self.line_width,
            "lines.linewidth": self.line_width * 1.5,
            "xtick.major.width": self.line_width,
            "ytick.major.width": self.line_width,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }


@dataclass
class PanelResult:
    panel: str
    status: str
    outputs: List[str] = field(default_factory=list)
    reason: str = ""


def add_style_args(parser) -> None:
    parser.add_argument("--width", type=float, default=None, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=None, help="Figure height in inches.")
    parser.add_argument("--font-size", type=float, default=7.0, help="Base font size (pt).")
    parser.add_argument("--line-width", type=float, default=0.8, help="Axis line width (pt).")
    parser.add_argument("--dpi", type=float, default=300.0, help="PNG preview resolution.")


def style_from_args(args, *, width: float, height: float) -> Style:
    return Style(
        width_in=args.width if args.width else width,
        height_in=args.height if args.height else height,
        font_size=args.font_size, line_width=args.line_width, dpi=args.dpi,
    )


# --- inputs --------------------------------------------------------------------

def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collected(data_dir: Path) -> Dict[str, Any]:
    """The files collect_fig3_data.py wrote, or a stated reason they are absent."""
    data_dir = Path(data_dir)
    meta = data_dir / "collected_metadata.json"
    if not meta.is_file():
        raise PanelSkip(MISSING_DATA, f"{meta} not found; run collect_fig3_data.py collect first.")
    out = {"dir": data_dir, "metadata": read_json(meta)}
    for name in ("runs", "samples", "published_rmse", "latency", "radar_metrics"):
        path = data_dir / f"{name}.csv"
        out[name] = read_csv(path) if path.is_file() else []
    radar_config = data_dir / "radar_config.json"
    out["radar_config"] = read_json(radar_config) if radar_config.is_file() else None
    return out


def model_order(meta: Dict[str, Any]) -> List[str]:
    return [str(m) for m in meta.get("model_order") or []]


def model_label(meta: Dict[str, Any], model: str) -> str:
    return str((meta.get("model_labels") or {}).get(model) or model)


def model_colors(order: Sequence[str]) -> Dict[str, str]:
    """Slot by position in model_order. More than eight is refused, never cycled."""
    if len(order) > len(CATEGORICAL):
        raise PanelSkip(
            INVALID_COMPARISON,
            f"model_order lists {len(order)} models; the validated palette has "
            f"{len(CATEGORICAL)} slots. Split the comparison into two figures.",
        )
    return {model: CATEGORICAL[i] for i, model in enumerate(order)}


def model_marker(order: Sequence[str], model: str) -> str:
    return MARKERS[list(order).index(model) % len(MARKERS)]


def to_float(text: Any) -> Optional[float]:
    if text is None or str(text).strip() in ("", "NA", "nan", "None", "not recorded"):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def is_true(text: Any) -> bool:
    return str(text).strip().lower() in ("true", "1", "yes")


def select_runs(runs: List[Dict[str, str]], *, allow_smoke: bool) -> tuple:
    """(kept, excluded_smoke). Smoke status comes from run.json only."""
    kept, excluded = [], []
    for row in runs:
        (kept if allow_smoke or not is_true(row.get("smoke")) else excluded).append(row)
    return kept, excluded


def kept_or_skip(rows: List[Dict[str, str]], *, allow_smoke: bool, what: str) -> tuple:
    """select_runs, raising MISSING_DATA when nothing survives the smoke filter."""
    if not rows:
        raise PanelSkip(MISSING_DATA, f"runs.csv holds no {what}.")
    kept, excluded = select_runs(rows, allow_smoke=allow_smoke)
    if not kept:
        raise PanelSkip(
            MISSING_DATA,
            f"all {len(excluded)} {what} are smoke runs (run.json smoke=true) and smoke "
            "runs are excluded by default; pass --allow-smoke to draw a "
            "workflow-validation example.",
        )
    return kept, excluded


def run_provenance(rows: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    """The per-run trace every panel's metadata carries."""
    keys = ("entry_id", "model", "dataset", "protocol", "seed", "smoke", "epochs_override",
            "run_id", "artifact", "artifact_sha256", "config_identity", "config_path",
            "label_annotator", "unit", "transform_status", "transform_chain",
            "stored_rows", "plotted_rows", "omitted_nonfinite", "n_cells")
    return [{"source_type": "local run", **{k: row.get(k) for k in keys}} for row in rows]


def unrecorded_smoke(rows: Iterable[Dict[str, str]]) -> List[str]:
    return [row["entry_id"] for row in rows
            if str(row.get("smoke")).strip().lower() not in ("true", "false")]


def column_label(dataset: str, protocol: str) -> str:
    return dataset if protocol in ("", dataset) else f"{dataset}\n{protocol}"


def fmt_value(value: Optional[float]) -> str:
    """Display text only; CSVs keep the full value."""
    if value is None:
        return "NA"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def first_seen(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


#: Fields that must agree for two runs to sit in one dataset/protocol column.
COLUMN_KEYS = ("schema", "label_annotator", "unit", "split_class", "sample_weighting",
               "soh_reference")


def check_columns(rows: List[Dict[str, str]]) -> None:
    """INVALID_COMPARISON when one dataset/protocol column mixes definitions."""
    columns: Dict[tuple, List[Dict[str, str]]] = {}
    for row in rows:
        columns.setdefault((row["dataset"], row["protocol"]), []).append(row)
    for (dataset, protocol), members in columns.items():
        for key in COLUMN_KEYS:
            values = sorted({str(m.get(key)) for m in members})
            if len(values) > 1:
                raise PanelSkip(
                    INVALID_COMPARISON,
                    f"column {dataset}/{protocol} mixes {key} values {values} "
                    f"(entries {[m['entry_id'] for m in members]}); runs with different "
                    "label definitions, units, splits or weighting are not compared.",
                )
    units = sorted({str(r.get("unit")) for r in rows})
    if len(units) > 1:
        raise PanelSkip(INVALID_COMPARISON, f"selected runs use different units {units}.")


def group_runs(rows: List[Dict[str, str]]) -> Dict[tuple, List[Dict[str, str]]]:
    """(model, dataset, protocol) -> runs. Refuses mixed configs and repeated seeds."""
    groups: Dict[tuple, List[Dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["dataset"], row["protocol"]), []).append(row)
    for key, members in groups.items():
        configs = sorted({str(m.get("config_identity")) for m in members})
        if len(configs) > 1:
            raise PanelSkip(
                INVALID_COMPARISON,
                f"{key[0]} on {key[1]}/{key[2]} has runs from {len(configs)} different "
                f"configurations ({configs}); different hyperparameters are not averaged "
                "into one model. Select one configuration, or give each its own model name.",
            )
        seeds = [str(m.get("seed")) for m in members]
        if len(seeds) != len(set(seeds)) or "not recorded" in seeds or "NA" in seeds:
            raise PanelSkip(
                INVALID_COMPARISON,
                f"{key[0]} on {key[1]}/{key[2]}: seeds {seeds} are repeated or not "
                "recorded, so the runs are not known to be independent replicates.",
            )
    return groups


def summarize(values: List[float]) -> Dict[str, Any]:
    """Mean, sample SD (n>=2 only) and n over independent runs."""
    return {"mean": sum(values) / len(values) if values else None,
            "sd": sample_sd(values), "n": len(values)}


def sample_sd(values: Sequence[float]) -> Optional[float]:
    """Sample SD (n-1). None for fewer than two independent runs."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


# --- outputs -------------------------------------------------------------------

def code_version() -> Dict[str, str]:
    from batteryml_web.viz.result_export import _git_state, _versions

    return {**_git_state(), **_versions()}


def command_line(argv: Optional[Sequence[str]], module: str) -> str:
    args = list(argv) if argv is not None else sys.argv[1:]
    return " ".join(["python", "-m", module] + [shlex.quote(a) for a in args])


def ensure_output(out_dir: Path, stems: Iterable[str]) -> None:
    """Refuse to overwrite an earlier panel output."""
    out_dir = Path(out_dir)
    for stem in stems:
        for suffix in (".svg", ".png", "_source_data.csv", "_metadata.json"):
            if (out_dir / f"{stem}{suffix}").exists():
                raise FileExistsError(
                    f"{out_dir / (stem + suffix)} already exists. Use a new output "
                    "directory; earlier outputs are never overwritten."
                )


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: ("NA" if v is None else repr(v) if isinstance(v, float) else v)
                for k, v in row.items()
            })


def save_panel(fig, out_dir: Path, stem: str, style: Style,
               rows: List[Dict[str, Any]], columns: List[str],
               metadata: Dict[str, Any]) -> List[str]:
    """Write <stem>.svg, .png, _source_data.csv and _metadata.json."""
    import matplotlib

    from batteryml_web.viz.render import render_to_bytes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_output(out_dir, [stem])
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        svg = render_to_bytes(fig, fmt="svg", dpi=style.dpi)
        png = render_to_bytes(fig, fmt="png", dpi=style.dpi)
    (out_dir / f"{stem}.svg").write_bytes(svg)
    (out_dir / f"{stem}.png").write_bytes(png)
    write_csv(out_dir / f"{stem}_source_data.csv", rows, columns)
    metadata = dict(metadata)
    metadata.setdefault("figure_style", {
        "width_in": style.width_in, "height_in": style.height_in,
        "font_size_pt": style.font_size, "line_width_pt": style.line_width,
        "png_dpi": style.dpi, "svg_text": "editable (svg.fonttype=none, DejaVu Sans; no font embedded)",
        "palette": "validated 8-slot categorical, slot = position in model_order",
    })
    metadata.setdefault("code_version", code_version())
    (out_dir / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    fig.clear()
    return [str(out_dir / f"{stem}{s}") for s in (".svg", ".png", "_source_data.csv", "_metadata.json")]


def base_parser(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory written by collect_fig3_data.py collect.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory; existing panel files are never overwritten.")
    parser.add_argument("--allow-smoke", action="store_true",
                        help="Include smoke runs; the panel is then labelled a workflow-validation example.")
    add_style_args(parser)
    return parser


def report(result_or_skip) -> int:
    """Print one panel's status line; exit code 0 = created, 2 = not drawn."""
    if isinstance(result_or_skip, PanelSkip):
        print(f"{result_or_skip.status}: {result_or_skip.reason}")
        return 2
    print(f"{result_or_skip.status}: " + ", ".join(result_or_skip.outputs))
    return 0


def smoke_annotation(fig, style: Style, smoke_used: bool) -> None:
    """A visible line on the panel when any smoke run was drawn."""
    if smoke_used:
        fig.text(0.01, 0.005, SMOKE_NOTE, fontsize=style.font_size - 1.5,
                 color=TEXT_SECONDARY, ha="left", va="bottom")


def new_figure(style: Style):
    """A detached Figure (no pyplot global state) under the panel's rc style."""
    from matplotlib.figure import Figure

    return Figure(figsize=(style.width_in, style.height_in), dpi=style.dpi)
