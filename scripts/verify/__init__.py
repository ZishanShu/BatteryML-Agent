"""Independent, fast, data-optional verification checks for BatteryML.

Every check in this package:

* is importable and runnable on its own (``python -m scripts.verify.check_environment``);
* returns a structured :class:`~scripts.verify.common.CheckResult`
  (check id, status, message, evidence) instead of printing free text;
* never raises to the top level — an unexpected exception becomes a FAIL
  result carrying the traceback as evidence;
* works with no data present, reporting SKIP with a reason rather than
  crashing;
* uses the standard library plus dependencies already in requirements.txt.

Run everything with ``python -m scripts.verify.run_all``.
"""
