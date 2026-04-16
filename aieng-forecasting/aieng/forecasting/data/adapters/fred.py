"""FRED (Federal Reserve Economic Data) adapter for the SeriesStore.

``FREDAdapter`` fetches a single FRED series and returns it in the canonical
internal format understood by :class:`~aieng.forecasting.data.store.SeriesStore`.

FRED data is deterministic and publicly available.  It is treated as offline
data: run ``scripts/fetch_fred.py`` before sessions to populate the local
cache.  No outbound calls occur during backtests or bootcamp sessions.

**API key requirement:** FRED requires a free API key obtained from
https://fred.stlouisfed.org/docs/api/api_key.html.  Provide it via the
``FRED_API_KEY`` environment variable (recommended) or the ``api_key``
constructor argument.

**``released_at`` approximation:** FRED does not expose vintage / release
dates through the standard ``fredapi`` interface.  The adapter sets
``released_at = timestamp``, which is correct for series that are available
at their reference period end (e.g. monthly averages published at or shortly
after month end).  For series with significant publication lags this is
optimistic and may be refined in a later pass using FRED's ``get_series_vintage_dates``
endpoint.
"""

from __future__ import annotations

import os

import pandas as pd

from aieng.forecasting.data.adapters.base import BaseAdapter


class FREDAdapter(BaseAdapter):
    """Adapter that fetches a single FRED series.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. ``"CPIFABSL"`` or ``"EXCAUS"``.
    api_key : str or None
        FRED API key.  If ``None``, the value is read from the
        ``FRED_API_KEY`` environment variable.

    Raises
    ------
    ValueError
        If no API key is available at construction time.

    Examples
    --------
    >>> adapter = FREDAdapter("EXCAUS")          # uses FRED_API_KEY env var
    >>> df = adapter.fetch()
    >>> df.columns.tolist()
    ['timestamp', 'value', 'released_at']
    """

    def __init__(self, series_id: str, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("FRED_API_KEY")
        if not resolved_key:
            raise ValueError(
                "FRED API key not provided.  Set the FRED_API_KEY environment variable "
                "or pass api_key= to FREDAdapter."
            )
        self._series_id = series_id
        self._api_key = resolved_key

    @property
    def series_id(self) -> str:
        """FRED series identifier."""
        return self._series_id

    def fetch(self) -> pd.DataFrame:
        """Fetch the FRED series and return it in canonical format.

        Downloads the full available history for the series.  Missing values
        (FRED uses a sentinel of ``"."`` for unreported periods) are dropped.

        ``released_at`` is set equal to ``timestamp`` — see module docstring for
        the rationale and limitation.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp`` (datetime64[ns]), ``value`` (float64),
            ``released_at`` (datetime64[ns]).  Sorted ascending by
            ``timestamp``.  Index is a default RangeIndex.

        Raises
        ------
        RuntimeError
            If the FRED API request fails.
        """
        try:
            from fredapi import Fred  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "fredapi is not installed. Run `uv add fredapi` to install it."
            ) from exc

        fred = Fred(api_key=self._api_key)

        try:
            raw: pd.Series = fred.get_series(self._series_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch FRED series '{self._series_id}': {exc}"
            ) from exc

        if raw.empty:
            raise RuntimeError(f"FRED series '{self._series_id}' returned no data.")

        df = raw.reset_index()
        df.columns = pd.Index(["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df["released_at"] = df["timestamp"]
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df[["timestamp", "value", "released_at"]]

    def __repr__(self) -> str:
        """Return a short representation without exposing the API key."""
        return f"FREDAdapter(series_id={self._series_id!r})"
