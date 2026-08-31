"""Extremely Randomized Trees RUL predictor.

Extension model: registered into BatteryML's ``MODELS`` registry from
``batteryml_web/`` without modifying ``batteryml/`` (`CLAUDE.md` §4). See
``docs/EXTENDING_MODELS.md``.

Sibling of the existing ``RandomForestRULPredictor``. Both are bagged ensembles
of unpruned trees; Extra Trees additionally randomizes the split thresholds
instead of optimizing them, which lowers variance further and trains faster at
the cost of a small increase in bias.

No hyperparameters are declared, here or in the configs, so
``ExtraTreesRegressor`` defaults apply — 100 trees, unlimited depth, all
features considered per split. Any value declared under ``model:`` in a config
is forwarded verbatim, so tuning is a config change and needs no edit here.
"""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor

from batteryml.builders import MODELS
from batteryml.models.sklearn_model import SklearnModel


@MODELS.register()
class ExtraTreesRULPredictor(SklearnModel):
    def __init__(self, workspace: str = None, **kwargs):
        SklearnModel.__init__(self, workspace)
        self.model = ExtraTreesRegressor(**kwargs)
