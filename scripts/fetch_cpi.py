"""Fetch and cache Canada-wide CPI series from Statistics Canada.

This script downloads table 18-10-0004-13 (Consumer Price Index by product
group, monthly, 2002=100 baseline) from Statistics Canada, filters to
Canada-wide series, and registers them in a DataService instance for
validation. The raw data is cached locally by the stats-can library in
``data/statcan/``.

Run this script once before starting a session or backtest to populate the
local cache. Re-running is safe and idempotent — the stats-can library skips
downloads when the cache is current.

Usage
-----
    uv run python scripts/fetch_cpi.py

Output
------
Prints a summary table of all registered series (series_id, date range,
number of observations).

Notes
-----
TODO: Consider generalizing this script for more flexible, user-friendly discovery. The StatCan
adapter is flexible, but callers must know the table id and member filters for the series they
want. We could add examples for a few different tables.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure the workspace root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import StatCanAdapter


# Statistics Canada table containing Canada-wide CPI by product group.
# Table 18-10-0004-13: Consumer Price Index by product group, monthly,
# not seasonally adjusted (2002=100 baseline).
CPI_TABLE_ID = "18-10-0004-13"

# Local directory where stats-can caches downloaded tables.
CACHE_DIR = Path("data/statcan")

# Canada-wide CPI series to register.
# Each entry: (series_id, product_group_label, description)
CPI_SERIES: list[tuple[str, str, str]] = [
    (
        "cpi_all_items_canada",
        "All-items",
        "CPI All-items, Canada (2002=100)",
    ),
    (
        "cpi_food_canada",
        "Food",
        "CPI Food, Canada (2002=100)",
    ),
    (
        "cpi_shelter_canada",
        "Shelter",
        "CPI Shelter, Canada (2002=100)",
    ),
    (
        "cpi_household_operations_canada",
        "Household operations, furnishings and equipment",
        "CPI Household operations, furnishings and equipment, Canada (2002=100)",
    ),
    (
        "cpi_clothing_canada",
        "Clothing and footwear",
        "CPI Clothing and footwear, Canada (2002=100)",
    ),
    (
        "cpi_transportation_canada",
        "Transportation",
        "CPI Transportation, Canada (2002=100)",
    ),
    (
        "cpi_health_personal_canada",
        "Health and personal care",
        "CPI Health and personal care, Canada (2002=100)",
    ),
    (
        "cpi_recreation_canada",
        "Recreation, education and reading",
        "CPI Recreation, education and reading, Canada (2002=100)",
    ),
    (
        "cpi_alcoholic_tobacco_canada",
        "Alcoholic beverages, tobacco products and recreational cannabis",
        "CPI Alcoholic beverages, tobacco and cannabis, Canada (2002=100)",
    ),
]


def build_data_service() -> DataService:
    """Build and populate a DataService with Canada-wide CPI series.

    Returns
    -------
    DataService
        DataService instance with all CPI series registered.
    """
    svc = DataService()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching StatCan table {CPI_TABLE_ID} → cache: {CACHE_DIR.resolve()}")
    print()

    succeeded = 0
    failed = 0

    for series_id, product_group, description in CPI_SERIES:
        adapter = StatCanAdapter(
            table_id=CPI_TABLE_ID,
            member_filter={
                "GEO": "Canada",
                "Products and product groups": product_group,
            },
            cache_dir=CACHE_DIR,
        )
        metadata = SeriesMetadata(
            series_id=series_id,
            description=description,
            source="StatCan",
            units="Index 2002=100",
            frequency="MS",
            table_id=CPI_TABLE_ID,
        )
        try:
            svc.register(series_id, adapter, metadata)
            succeeded += 1
        except Exception as exc:
            print(f"  [WARN] Failed to register {series_id!r}: {exc}")
            failed += 1

    print(f"Registered {succeeded} series ({failed} failed).")
    return svc


def main() -> None:
    """Fetch CPI data and print a summary."""
    svc = build_data_service()

    print()
    summary = svc.summary()
    if summary.empty:
        print("No series registered.")
        return

    # Format for display.
    summary["start"] = summary["start"].dt.strftime("%Y-%m")
    summary["end"] = summary["end"].dt.strftime("%Y-%m")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
