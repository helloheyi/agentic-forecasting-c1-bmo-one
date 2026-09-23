"""Shared lock guarding Darts model construction across threads.

Every Darts forecasting model (``ExponentialSmoothing``, ``KalmanForecaster``,
``AutoARIMA``, ``LinearRegressionModel``, ``LightGBMModel``, ...) is built with
a ``ModelMeta`` metaclass (``darts/models/forecasting/forecasting_model.py``)
that stashes constructor arguments on the **class** (not the instance) as a
hand-off from ``__call__`` to ``__init__``::

    cls._model_call = all_params   # __call__
    ...
    model_params = copy.deepcopy(self._model_call)   # __init__
    del self.__class__._model_call

This is not thread-safe: constructing two instances of the *same* Darts model
class concurrently (e.g. two origins' ``ExponentialSmoothing(...)`` calls
racing inside a thread pool — see
:func:`aieng.forecasting.evaluation.backtest.run_eval_loop`'s ``n_jobs``) can
raise ``AttributeError: '<Model>' object has no attribute '_model_call'``, or
in principle let one construction silently pick up another's arguments,
depending on interleaving. The race is scoped to the brief constructor
call — not ``.fit()``/``.predict()`` — so every Darts model construction
across every predictor wrapper in this package shares this one lock to
serialize just that hand-off, while the actual fit/predict work (where
origin-level and predictor-level parallelism's real benefit lives) still runs
concurrently. See ``docs/backtest-parallelism.md`` for the full incident.
"""

from __future__ import annotations

import threading

#: Serializes every Darts model constructor call across this process. A single
#: shared instance is required — a per-file or per-predictor lock would not
#: protect against two *different* predictors (e.g. ``lightgbm`` and
#: ``lightgbm_cov``) racing to construct the same underlying Darts model class
#: concurrently under Section 5's predictor-level parallelism.
DARTS_MODEL_CONSTRUCTION_LOCK = threading.Lock()
