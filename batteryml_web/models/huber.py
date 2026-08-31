"""Huber (robust linear) RUL predictor.

Extension model: registered into BatteryML's ``MODELS`` registry from
``batteryml_web/`` without modifying ``batteryml/`` (`CLAUDE.md` §4). See
``docs/EXTENDING_MODELS.md``.

The only robust regressor in the grid. Ridge, Lasso and ElasticNet all minimize
squared error, so a handful of cells that fail unusually early or late pull the
fit towards themselves in proportion to the square of their residual. The Huber
loss is quadratic for small residuals and linear beyond a threshold, so those
cells are down-weighted instead. Comparing it against Ridge on the same split
is therefore an indication of how much a result depends on outlier cells.

Two properties worth knowing before reading its numbers:

* ``HuberRegressor`` applies an L2 penalty of its own (``alpha=0.0001`` by
  default), so it is not an unpenalized fit.
* It solves iteratively with ``max_iter=100`` by default and can emit
  ``ConvergenceWarning`` without failing. A warning is not an error, but a run
  that produces one has not converged and its coefficients should not be read
  as final. Raise ``max_iter`` in the config if that happens; it is forwarded
  verbatim.

No hyperparameters are declared, so those defaults apply.
"""

from __future__ import annotations

from sklearn.linear_model import HuberRegressor

from batteryml.builders import MODELS
from batteryml.models.sklearn_model import SklearnModel


@MODELS.register()
class HuberRULPredictor(SklearnModel):
    def __init__(self, workspace: str = None, **kwargs):
        SklearnModel.__init__(self, workspace)
        self.model = HuberRegressor(**kwargs)
