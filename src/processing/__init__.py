"""
Data processing modules for MakeSenseOfIt.
Handles deduplication, aggregation, and analysis of collected posts.
"""
from .deduplicator import Deduplicator
from .aggregator import DataAggregator
from .time_series import TimeSeriesAnalyzer

__all__ = ['Deduplicator', 'DataAggregator', 'TimeSeriesAnalyzer']