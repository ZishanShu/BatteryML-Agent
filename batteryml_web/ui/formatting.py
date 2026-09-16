"""Display formatting that never becomes the value.

The result page showed metrics at dashboard scale, where a 16-significant-
digit float clipped to "455.58532714…" — big, and unreadable. Shortening the
*display* is right; shortening the value is not. Everything here returns a
string for the page, leaves the canonical number untouched in the result
context, and keeps the full-precision form available beside it.

No Streamlit import: these are pure functions, so "formatting does not alter
the canonical value" is a unit test.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Significant decimals in a compact metric. Enough to compare two runs at a
#: glance, few enough to fit without clipping.
METRIC_DECIMALS = 3

#: Decimals in the supplementary MAPE percentage.
PERCENT_DECIMALS = 2


def is_real_number(value: Any) -> bool:
    """True for a finite int/float. ``bool`` is not a metric."""
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def compact_metric(value: Any) -> str:
    """A metric at reading scale, e.g. ``455.585``.

    Falls back to the exact repr for anything that is not a finite number, so
    an unexpected value is visible rather than silently formatted into
    something plausible.
    """
    if not is_real_number(value):
        return str(value)
    return f"{float(value):.{METRIC_DECIMALS}f}"


def full_precision(value: Any) -> str:
    """The canonical value as the backend holds it, for a details view."""
    return repr(value) if not is_real_number(value) else repr(float(value))


def mape_percent(value: Any) -> Optional[str]:
    """MAPE as a percentage, display only.

    The canonical MAPE stays the ratio the evaluator computed; this is the
    reading most people want next to it. Returns None when the value is not
    a number, so the page shows nothing rather than "nan%".
    """
    if not is_real_number(value):
        return None
    return f"{float(value) * 100:.{PERCENT_DECIMALS}f}%"


def metric_display(metrics: Optional[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Everything the page needs about one metric, without changing it.

    Returns ``{"present", "compact", "exact", "percent"}``. ``percent`` is
    filled only for MAPE. ``exact`` is the canonical value's own repr, so a
    details view can show what the artifact holds.
    """
    if not isinstance(metrics, dict) or key not in metrics:
        return {"present": False, "compact": None, "exact": None, "percent": None}
    value = metrics[key]
    return {
        "present": True,
        "compact": compact_metric(value),
        "exact": full_precision(value),
        "percent": mape_percent(value) if key.upper() == "MAPE" else None,
    }
