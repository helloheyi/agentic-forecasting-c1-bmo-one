"""Fetch and cache FRED economic series relevant to food price forecasting.

This script downloads a curated set of FRED (Federal Reserve Economic Data)
series that serve as exogenous covariates for the Canada's Food Price Report
(CFPR) forecasting experiment.  It registers them in a ``DataService`` for
validation and prints a summary of what was fetched.

The series are cached in memory only during this validation run; actual
experiment notebooks create their own ``DataService`` and call ``FREDAdapter``
directly, using the FRED REST API each time (no local FRED cache exists at
this stage — see technical-design.md for the planned ``FREDAdapter``
caching extension).

**Prerequisite:** Set ``FRED_API_KEY`` in your environment (or in ``.env``)::

    export FRED_API_KEY=your_key_here

Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html.

Usage
-----
    uv run python scripts/fetch_fred.py

Output
------
Prints a summary table of all registered FRED series (series_id, date range,
number of observations).
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import FREDAdapter


# ---------------------------------------------------------------------------
# FRED series catalogue for food price forecasting
#
# Each entry: (series_id, fred_series_id, description, units)
#
# Rationale for inclusion:
#   - US food CPI sub-indices: US prices transmit to Canadian food costs
#     through trade and supply chains, especially for commodities.
#   - Canadian 10-year bond yield: measures cost of capital and credit
#     conditions affecting food production and distribution.
#   - Canada/US exchange rate: direct pass-through to import food prices.
#   - S&P 100 Volatility (VXO): broad proxy for financial market uncertainty,
#     correlated with commodity price volatility.
#   - Wilshire 5000: broad US equity index as a leading economic indicator.
# ---------------------------------------------------------------------------

FRED_SERIES: list[tuple[str, str, str, str]] = [
    (
        "fred_us_cpi_food_at_home",
        "CPIFABSL",
        "US CPI: Food at Home, All Urban Consumers (1982-84=100)",
        "Index 1982-84=100",
    ),
    (
        "fred_us_cpi_meats_poultry_fish_eggs",
        "CUSR0000SAF112",
        "US CPI: Meats, Poultry, Fish, and Eggs, All Urban Consumers (1982-84=100)",
        "Index 1982-84=100",
    ),
    (
        "fred_us_cpi_fruits_vegetables",
        "CUSR0000SAF113",
        "US CPI: Fruits and Vegetables, All Urban Consumers (1982-84=100)",
        "Index 1982-84=100",
    ),
    (
        "fred_canada_10yr_bond_yield",
        "IRLTLT01CAM156N",
        "Canada Long-Term Government Bond Yields: 10-Year (% per annum)",
        "Percent per annum",
    ),
    (
        "fred_canada_us_exchange_rate",
        "EXCAUS",
        "Canada / US Foreign Exchange Rate (CAD per 1 USD, monthly average)",
        "CAD per USD",
    ),
    (
        "fred_sp100_volatility_vxo",
        "VXOCLS",
        "CBOE S&P 100 Volatility Index (VXO), daily close (monthly avg in FRED)",
        "Index",
    ),
    (
        "fred_wilshire_5000",
        "WILL5000IND",
        "Wilshire 5000 Total Market Full Cap Index",
        "Index",
    ),
]


def build_data_service() -> DataService:
    """Build and populate a DataService with FRED series.

    Returns
    -------
    DataService
        DataService instance with all FRED series registered.
    """
    svc = DataService()
    print("Fetching FRED series (requires FRED_API_KEY)...")
    print()

    succeeded = 0
    failed = 0

    for series_id, fred_id, description, units in FRED_SERIES:
        adapter = FREDAdapter(fred_id)
        metadata = SeriesMetadata(
            series_id=series_id,
            description=description,
            source=f"FRED ({fred_id})",
            units=units,
            frequency="MS",
        )
        try:
            svc.register(series_id, adapter, metadata)
            succeeded += 1
            print(f"  ✓ {series_id} ({fred_id})")
        except Exception as exc:
            print(f"  ✗ {series_id} ({fred_id}): {exc}")
            failed += 1

    print()
    print(f"Registered {succeeded} series ({failed} failed).")
    return svc


def main() -> None:
    """Fetch FRED data and print a summary."""
    svc = build_data_service()

    print()
    summary = svc.summary()
    if summary.empty:
        print("No series registered.")
        return

    summary["start"] = summary["start"].dt.strftime("%Y-%m")
    summary["end"] = summary["end"].dt.strftime("%Y-%m")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
