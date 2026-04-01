"""Adapter implementations for ingesting data into the SeriesStore."""

from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.data.adapters.statcan import StatCanAdapter


__all__ = ["BaseAdapter", "StatCanAdapter"]
