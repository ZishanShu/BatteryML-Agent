"""Browser viewport audit: no horizontal page overflow, one type scale, every page.

Opens a running BatteryML web app in headless Chromium at 1440, 1024 and
760px wide, visits every page through the sidebar, opens the sidebar LLM
Settings, and records for each view:

* horizontal overflow of the document and of the main column, in px
  (``scrollWidth - clientWidth``; 0 means no horizontal page scroll);
* whether Streamlit showed an exception;
* the computed size and weight of every heading, the font family, and the
  radius of the first input, select and button.

It reads the page and clicks navigation only. It never presses Run, Connect
or any other action, and it does not change data, configs or settings.

Optional tooling — not a dependency of the platform. It needs Playwright::

    pip install playwright==1.63.0
    python -m playwright install chromium

Usage (from the repository root, with the app already running)::

    python -m streamlit run batteryml_web/app.py --server.port 8501 --server.headless true &
    python scripts/ui_audit/viewport_audit.py --url http://localhost:8501

Screenshots and ``viewport_audit.json`` go to ``--out-dir`` (default
``runs/ui_audit``, which git ignores). Exit status: 0 when no view overflows
and none shows an exception; 1 otherwise; 2 when Playwright is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

WIDTHS = (1440, 1024, 760)
PAGES = ("home", "agent_workflow", "prepare_data", "run_model", "review_results")
HOME_READY_TEXT = "Quick experience"

MEASURE_JS = r"""
() => {
  const doc = document.documentElement;
  const main = document.querySelector('[data-testid="stMain"]');
  const style = (el, props) => {
    if (!el) return null;
    const s = getComputedStyle(el);
    return Object.fromEntries(props.map(p => [p, s.getPropertyValue(p)]));
  };
  const first = (sel, props) => style(document.querySelector(sel), props);
  const headings = Array.from(main ? main.querySelectorAll('h1, h2, h3, h4') : [])
    .filter(h => h.offsetParent !== null)
    .map(h => ({tag: h.tagName, text: (h.innerText || '').trim().slice(0, 48),
                size: getComputedStyle(h).fontSize, weight: getComputedStyle(h).fontWeight}));
  return {
    overflow_document_px: doc.scrollWidth - doc.clientWidth,
    overflow_main_px: main ? main.scrollWidth - main.clientWidth : null,
    exception_shown: !!document.querySelector('[data-testid="stException"]')
                     || document.body.innerText.includes('Something went wrong'),
    body_font_family: first('[data-testid="stMain"] p', ['font-family']),
    headings: headings,
    input: first('[data-testid="stMain"] [data-testid="stTextInput"] input', ['border-radius', 'border-top-color']),
    select: first('[data-testid="stMain"] [data-testid="stSelectbox"] [role="group"]', ['border-radius', 'border-top-color']),
    button: first('[data-testid="stMain"] [data-testid="stButton"] > button', ['border-radius', 'font-weight']),
    settings_input: first('[data-testid="stSidebar"] .st-key-sb_foot [data-testid="stTextInput"] input',
                          ['border-radius', 'border-top-color', 'background-color']),
  };
}
"""


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="URL of the running app, e.g. http://localhost:8501")
    parser.add_argument("--out-dir", default="runs/ui_audit", help="where screenshots and the JSON report go")
    parser.add_argument("--chromium", default=None,
                        help="path to a Chromium executable (default: Playwright's own)")
    parser.add_argument("--widths", default=",".join(str(w) for w in WIDTHS),
                        help="comma-separated viewport widths in px")
    parser.add_argument("--settle-seconds", type=float, default=4.0,
                        help="wait after each navigation for Streamlit to finish rendering")
    return parser.parse_args(argv)


def _go(page, name: str, settle: float) -> None:
    """Navigate with the sidebar entry, exactly as a user would."""
    button = page.locator(f".st-key-sidebar_go_{name} button")
    if button.count() == 0:
        raise RuntimeError(f"sidebar entry 'sidebar_go_{name}' not found; is {name!r} still a page?")
    button.first.evaluate("el => el.click()")
    time.sleep(settle)


def audit(url: str, out_dir: Path, widths: List[int], chromium: Optional[str], settle: float) -> Dict[str, dict]:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, dict] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chromium) if chromium else pw.chromium.launch()
        try:
            for width in widths:
                page = browser.new_page(viewport={"width": width, "height": 1200})
                page.goto(url, wait_until="networkidle")
                page.get_by_text(HOME_READY_TEXT, exact=True).first.wait_for(timeout=120_000)
                time.sleep(settle)
                for name in PAGES:
                    if name != "home":
                        _go(page, name, settle)
                    page.screenshot(path=str(out_dir / f"w{width}_{name}.png"), full_page=True)
                    report[f"w{width}/{name}"] = page.evaluate(MEASURE_JS)
                summary = page.locator('[data-testid="stSidebar"] summary', has_text="Settings")
                if summary.count():
                    summary.first.click()
                    time.sleep(settle / 2)
                    page.screenshot(path=str(out_dir / f"w{width}_settings.png"), full_page=True)
                    report[f"w{width}/settings"] = page.evaluate(MEASURE_JS)
                page.close()
        finally:
            browser.close()
    return report


def failures(report: Dict[str, dict]) -> List[str]:
    """Views with a horizontal page scroll or an exception on screen."""
    bad = []
    for view, measured in report.items():
        if measured["overflow_document_px"] > 0 or (measured["overflow_main_px"] or 0) > 0:
            bad.append(f"{view}: horizontal overflow document={measured['overflow_document_px']}px "
                       f"main={measured['overflow_main_px']}px")
        if measured["exception_shown"]:
            bad.append(f"{view}: Streamlit shows an exception")
    return bad


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        import playwright  # noqa: F401
    except ModuleNotFoundError:
        print("Playwright is not installed. It is optional tooling for this audit only:\n"
              "    pip install playwright==1.63.0\n    python -m playwright install chromium",
              file=sys.stderr)
        return 2
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    out_dir = Path(args.out_dir)
    report = audit(args.url, out_dir, widths, args.chromium, args.settle_seconds)
    (out_dir / "viewport_audit.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for view, measured in report.items():
        sizes = " ".join(f"{h['tag']}:{h['size']}" for h in measured["headings"][:6])
        print(f"{view:<24} overflow={measured['overflow_document_px']}/{measured['overflow_main_px']}px "
              f"exception={measured['exception_shown']} headings={sizes}")
    bad = failures(report)
    print("\n".join(["", "FAIL:"] + bad) if bad else "\nPASS: no horizontal page overflow, no exception")
    print(f"Screenshots and viewport_audit.json: {out_dir}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
