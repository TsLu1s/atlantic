"""Stand-in for the ``h2o`` package, used by tests that exercise the
H2O feature-selection code path without a JVM."""

from __future__ import annotations

import types

import pandas as pd


def importance_frame(rows):
    """Build a frame shaped like ``H2OModel.varimp(use_pandas=True)``."""
    return pd.DataFrame(
        rows, columns=["variable", "relative_importance", "scaled_importance", "percentage"]
    )


def make_h2o_stub(importances, init_error=None):
    """Minimal stand-in for the ``h2o`` package.

    Args:
        importances: Mapping of model id -> varimp DataFrame (or None), in
            leaderboard order.
        init_error: Exception raised by ``h2o.init()`` (simulates a missing
            Java runtime).

    Returns:
        ``sys.modules`` entries to patch in: ``{"h2o": ..., "h2o.automl": ...}``.
        The ``h2o`` module records ``removed`` keys, ingested ``frames``,
        ``automl_kwargs`` and ``trained_on``.
    """
    h2o = types.ModuleType("h2o")
    automl = types.ModuleType("h2o.automl")
    h2o.removed, h2o.frames, h2o.automl_kwargs = [], [], {}

    class Column:
        def asfactor(self):
            return self

    class H2OFrame:
        frame_id = "frame_key"

        def __init__(self, df):
            h2o.frames.append(df)

        def __getitem__(self, key):
            return Column()

        def __setitem__(self, key, value):
            pass

    class Model:
        def __init__(self, model_id):
            self.model_id = model_id

        def varimp(self, use_pandas=True):
            return importances[self.model_id]

    class Leaderboard:
        def as_data_frame(self):
            return pd.DataFrame({"model_id": list(importances)})

    class H2OAutoML:
        def __init__(self, **kwargs):
            h2o.automl_kwargs.update(kwargs)
            self.leaderboard = Leaderboard()

        def train(self, x, y, training_frame):
            h2o.trained_on = (list(x), y)

    def init():
        if init_error is not None:
            raise init_error

    h2o.init = init
    h2o.H2OFrame = H2OFrame
    h2o.get_model = Model
    h2o.remove = h2o.removed.append
    automl.H2OAutoML = H2OAutoML
    h2o.automl = automl
    return {"h2o": h2o, "h2o.automl": automl}
