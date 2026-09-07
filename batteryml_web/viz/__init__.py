"""Figure generation for the BatteryML protocol platform.

Scope and boundaries
--------------------
This package **draws** figures. It never trains, never evaluates, and never
produces a metric of its own.

* Every aggregate metric a figure displays (RMSE / MAE / MAPE) is read verbatim
  from the ``scores`` dict that BatteryML's evaluator wrote into the prediction
  artifact. Nothing here recomputes one for display (``CLAUDE.md`` §3).
* The one place BatteryML's evaluator *is* called is
  :func:`~batteryml_web.viz.loaders.verify_recorded_scores`, and it is called as
  a **consistency gate**, not to produce a number: if the labels this package
  inverse-transformed do not reproduce the recorded scores, the figure would
  silently disagree with ``metrics.csv``, so loading fails loudly instead.
* Pointwise differences (``y_pred - y_true``) are computed here, because a
  residual plot is nothing but those differences. They are display quantities
  and are never written to ``metrics.csv`` or reported as a metric.

Layout
------
``style.py``    theme tokens (light / dark) and axis chrome. No data.
``loaders.py``  artifact -> validated arrays. Pure, no plotting, no I/O beyond
                reading the paths it is given.
``figures.py``  arrays -> ``matplotlib.figure.Figure``. Pure, no I/O, no global
                pyplot state.
``render.py``   the only module that writes files, plus the figure manifest.
``cli.py``      the offline entry point. Route B: no API key, no Streamlit,
                no network (``CLAUDE.md`` §1).

Spec IDs implemented here: ``VIZ-01``..``VIZ-04`` (see ``docs/SPEC.md``).
Verified by the ``FIG-01`` check (``scripts/verify/check_figure_contract.py``).
"""

from __future__ import annotations

from .style import DARK, LIGHT, THEMES, Theme, get_theme

__all__ = ["DARK", "LIGHT", "THEMES", "Theme", "get_theme"]
