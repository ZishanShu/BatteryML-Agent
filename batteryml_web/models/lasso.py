"""Lasso (L1-penalized linear) RUL predictor with a cross-validated penalty.

Extension model: registered into BatteryML's ``MODELS`` registry from
``batteryml_web/`` without modifying ``batteryml/`` (`CLAUDE.md` §4). See
``docs/EXTENDING_MODELS.md``.

Completes the penalized-linear family already in the grid: Ridge is pure L2,
ElasticNet mixes L1 and L2, and this is pure L1 — the ``l1_ratio = 1`` corner
of the same family. Being pure L1 it drives coefficients to exactly zero, so it
is the one member that selects a subset of the feature dimensions rather than
shrinking all of them.

``LassoCV`` rather than ``Lasso``, mirroring the existing
``ElasticNetRULPredictor``, which wraps ``ElasticNetCV``: the penalty strength
is chosen by internal k-fold cross-validation on the training split, so no
alpha has to be declared in a config. Note this makes fitting itself
cross-validated — the model sees only training cells, but it consumes them in
folds.

No hyperparameters are declared, so ``LassoCV`` defaults apply — 100 candidate
alphas on an automatically chosen path, 5-fold CV. Any value declared under
``model:`` in a config is forwarded verbatim.
"""

from __future__ import annotations

from sklearn.linear_model import LassoCV

from batteryml.builders import MODELS
from batteryml.models.sklearn_model import SklearnModel


@MODELS.register()
class LassoRULPredictor(SklearnModel):
    def __init__(self, workspace: str = None, **kwargs):
        SklearnModel.__init__(self, workspace)
        self.model = LassoCV(**kwargs)
