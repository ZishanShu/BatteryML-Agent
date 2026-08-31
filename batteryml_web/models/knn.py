"""k-Nearest Neighbors RUL predictor.

Extension model: registered into BatteryML's ``MODELS`` registry from
``batteryml_web/`` without modifying ``batteryml/`` (`CLAUDE.md` §4). See
``docs/EXTENDING_MODELS.md``.

The only purely instance-based baseline in the grid: it fits nothing and
predicts by averaging the labels of the nearest training cells. That makes it a
useful reference point — a parametric model that cannot beat it is not learning
much from the feature representation.

Distance-based, so it depends on the feature scaling the configs already apply
through ``feature_transformation: ZScoreDataTransformation``. The one config
that does not declare a feature transformation (``mix.yaml``, inherited from
the template) therefore feeds it unstandardized features; that absence is a
blocked decision and is carried over rather than corrected here.

No hyperparameters are declared, so ``KNeighborsRegressor`` defaults apply —
5 neighbors, uniform weights, Euclidean distance. Any value declared under
``model:`` in a config is forwarded verbatim, so tuning is a config change and
needs no edit here.
"""

from __future__ import annotations

from sklearn.neighbors import KNeighborsRegressor

from batteryml.builders import MODELS
from batteryml.models.sklearn_model import SklearnModel


@MODELS.register()
class KNNRULPredictor(SklearnModel):
    def __init__(self, workspace: str = None, **kwargs):
        SklearnModel.__init__(self, workspace)
        self.model = KNeighborsRegressor(**kwargs)
