"""Preprocessing module — error aggregation, metrics, and log chunking.

Public API:
    ErrorAggregator   — rule-based error categorisation.
    ErrorRecord       — per-category error data model.
    MetricsGenerator  — runtime metrics from a BatchExecution.
    BatchMetrics      — metrics data model.
    LogChunker        — semantic log chunker.
    LogChunk          — chunk data model.
"""

from app.preprocessing.error_aggregator import ErrorAggregator, ErrorRecord
from app.preprocessing.metrics_generator import BatchMetrics, MetricsGenerator
from app.preprocessing.chunker import LogChunk, LogChunker

__all__ = [
    "ErrorAggregator",
    "ErrorRecord",
    "MetricsGenerator",
    "BatchMetrics",
    "LogChunker",
    "LogChunk",
]
