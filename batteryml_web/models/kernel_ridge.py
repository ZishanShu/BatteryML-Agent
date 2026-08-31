"""Kernel Ridge RUL predictor.

Extension model: registered into BatteryML's ``MODELS`` registry from
``batteryml_web/`` without modifying ``batteryml/`` (`CLAUDE.md` §4). See
``docs/EXTENDING_MODELS.md``.

Sits between the two kernel methods already in the grid. It shares its loss
with ``RidgeRULPredictor`` (squared error plus L2) and its kernel trick with
``SVMRULPredictor`` and ``GaussianProcessRULPredictor``, but unlike SVR it has
a closed-form solution and unlike GPR it produces no predictive variance. With
the ~40-100 training cells these splits provide, the N x N kernel matrix is
trivially small.

**The default kernel is ``'linear'``, not RBF.** Left undeclared deliberately:
choosing a kernel and its bandwidth determines the numbers this model produces,
and that is a modelling decision to make deliberately with a stated rationale,
not a default to inherit silently from a wrapper. Declare it in the config when
you want one, for example::

    model:
        name: 'KernelRidgeRULPredictor'
        kernel: 'rbf'
        gamma: 0.001

As declared, this is a linear model regularized like Ridge, so its numbers
should land near ``RidgeRULPredictor``'s rather than adding a new model family
to the comparison. Treat a nonlinear kernel as the interesting configuration.

Every ``model:`` key is forwarded verbatim to ``KernelRidge``, so this is a
config change and needs no edit here.
"""

from __future__ import annotations

from sklearn.kernel_ridge import KernelRidge

from batteryml.builders import MODELS
from batteryml.models.sklearn_model import SklearnModel


@MODELS.register()
class KernelRidgeRULPredictor(SklearnModel):
    def __init__(self, workspace: str = None, **kwargs):
        SklearnModel.__init__(self, workspace)
        self.model = KernelRidge(**kwargs)
