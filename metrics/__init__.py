"""
Astra Metrics Engine — Centralized performance measurement.

Single source of truth for all portfolio/strategy/risk metrics.
Every subsystem (paper_trader, truth_report, gates, dashboard) consumes
metrics from here instead of computing its own.
"""

from metrics.confidence import SampleGate, confidence_interval, sample_size_gate
from metrics.drawdown import DrawdownState, DrawdownTracker
from metrics.performance import MetricsSnapshot, PerformanceEngine, TradeRecord

__all__ = [
    "PerformanceEngine",
    "MetricsSnapshot",
    "TradeRecord",
    "DrawdownTracker",
    "DrawdownState",
    "confidence_interval",
    "sample_size_gate",
    "SampleGate",
]
