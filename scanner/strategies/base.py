"""
Abstract Strategy Base Class for Astra V2.

Inspired by:
  - jaredzwick/polymarket-trading-bot (TypeScript BaseStrategy)
  - pecan987/trading-bot-framework (Python abstract base)
  - WheelForge (Sharpe/drawdown/win-rate metrics tracking)

Every strategy in Astra is a concrete subclass of BaseStrategy.
Each strategy:
  1. Receives market data + context → returns TradeSignal | None
  2. Tracks its own performance metrics automatically (Sharpe, win rate, drawdown)
  3. Can be enabled/disabled without restarting the bot
  4. Produces a standardised metrics dict for the report dashboard

Existing strategies (to be refactored into subclasses):
  - LongshotStrategy     — systematic BUY NO on 1-8¢ YES markets
  - ArbitrageStrategy    — YES+NO combined < $0.97
  - LeaderFollowerStrategy — semantic cluster follower signals
  - AstraV2Strategy      — full adversarial AI estimation pipeline

New strategies can be added by:
  1. Subclassing BaseStrategy
  2. Implementing evaluate()
  3. Registering in STRATEGY_REGISTRY below

Confidence gate: signals with confidence < 0.5 are ignored by the engine.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import math

from scanner.market_fetcher import Market


# ─────────────────────────────────────────────────────────────────────────────
# Trade Signal — the standard output of every strategy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    """
    A trading signal produced by a strategy.
    confidence must be > 0.5 for the engine to act on it.
    """
    condition_id: str
    question: str
    direction: str              # "BUY YES" | "BUY NO"
    confidence: float           # 0.0–1.0 — must be >0.5 to execute
    target_price: float         # Price we expect to pay
    kelly_pct: float            # Suggested position size as % of bankroll
    edge: float                 # Our estimate - market price (signed)
    ev_after_costs: float       # Expected value after fees
    source: str                 # Strategy name
    reasoning: str              # Brief explanation
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def is_actionable(self) -> bool:
        """True if this signal meets minimum quality bar."""
        return (
            self.confidence > 0.5
            and self.ev_after_costs > 0
            and self.kelly_pct > 0
        )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Context — everything a strategy needs to evaluate a market
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyContext:
    """
    Passed to strategy.evaluate() on every call.
    Aggregates all available context: signals, prices, calendar events, whales.
    """
    market_context: object = None          # MarketContext from signals.py
    whale_signals: dict = field(default_factory=dict)   # {cid: WhaleSignal}
    calendar_events: dict = field(default_factory=dict) # {cid: [(event, hours_delta)]}
    price_data: dict = field(default_factory=dict)      # {coin_id: price}
    sports_estimates: dict = field(default_factory=dict)
    learning_context: str = ""
    vix_kelly_mult: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Performance Metrics Tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyMetrics:
    """
    Per-strategy performance tracking.
    Mirrors the metrics from WheelForge and jaredzwick BaseStrategy.
    """
    name: str
    total_signals: int = 0
    executed_signals: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    pnl_history: list[float] = field(default_factory=list)   # per-trade returns
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0

    def record_outcome(self, pnl: float):
        """Call when a trade from this strategy resolves."""
        self.pnl_history.append(pnl)
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        # Update peak and drawdown
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        drawdown = self.peak_pnl - self.total_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """
        Simplified Sharpe ratio from per-trade P&L history.
        Uses 0 as risk-free rate (binary outcomes have no 'risk-free' leg).
        """
        if len(self.pnl_history) < 4:
            return 0.0
        n = len(self.pnl_history)
        mean = sum(self.pnl_history) / n
        variance = sum((x - mean) ** 2 for x in self.pnl_history) / n
        std = math.sqrt(variance) if variance > 0 else 0.0
        return round(mean / std, 3) if std > 0 else 0.0

    def to_dict(self) -> dict:
        total = self.wins + self.losses
        return {
            "strategy": self.name,
            "signals": self.total_signals,
            "executed": self.executed_signals,
            "win_rate": f"{self.win_rate:.0%}",
            "sharpe": f"{self.sharpe_ratio:.2f}",
            "total_pnl": f"${self.total_pnl:+.2f}",
            "max_drawdown": f"${self.max_drawdown:.2f}",
            "trades": total,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base Strategy
# ─────────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """
    Abstract base class for all Astra trading strategies.

    Subclasses must implement evaluate().
    All other methods (metrics, enable/disable, lifecycle) are provided.
    """

    def __init__(self, name: str):
        self.name = name
        self._enabled = True
        self.metrics = StrategyMetrics(name=name)

    @abstractmethod
    def evaluate(
        self,
        market: Market,
        context: StrategyContext,
    ) -> Optional[TradeSignal]:
        """
        Evaluate a single market and return a TradeSignal or None.

        Rules:
          - Return None if there is no actionable signal
          - Signal confidence must be > 0.5 for the engine to act on it
          - Never force a trade; 'no signal' is always a valid response
          - Kelly % must already account for confidence and robustness
        """
        ...

    def evaluate_batch(
        self,
        markets: list[Market],
        context: StrategyContext,
    ) -> list[TradeSignal]:
        """
        Evaluate all markets. Filters disabled strategies and low-confidence signals.
        Subclasses can override for efficiency (e.g. vectorised screening).
        """
        if not self._enabled:
            return []

        signals = []
        for market in markets:
            try:
                signal = self.evaluate(market, context)
                if signal is not None:
                    self.metrics.total_signals += 1
                    if signal.is_actionable:
                        signals.append(signal)
            except Exception:
                pass   # Strategy errors never crash the scan loop

        return signals

    def on_trade_executed(self, signal: TradeSignal):
        """Called when this strategy's signal is actually executed."""
        self.metrics.executed_signals += 1

    def on_outcome(self, condition_id: str, pnl: float):
        """Called when a trade from this strategy resolves. Updates metrics."""
        self.metrics.record_outcome(pnl)

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def get_metrics(self) -> dict:
        return self.metrics.to_dict()

    def __repr__(self) -> str:
        status = "ON" if self._enabled else "OFF"
        return f"<{self.__class__.__name__} '{self.name}' [{status}]>"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Registry
# ─────────────────────────────────────────────────────────────────────────────

class StrategyRegistry:
    """
    Central registry of all active strategies.
    Replaces the ad-hoc calls in paper_trader.py's run_paper_scan().

    Usage:
        registry = StrategyRegistry()
        registry.register(LongshotStrategy())
        registry.register(ArbitrageStrategy())
        signals = registry.run_all(markets, context)
    """

    def __init__(self):
        self._strategies: list[BaseStrategy] = []

    def register(self, strategy: BaseStrategy):
        self._strategies.append(strategy)

    def run_all(
        self,
        markets: list[Market],
        context: StrategyContext,
    ) -> list[TradeSignal]:
        """Run all enabled strategies and return combined signals, deduped by market."""
        all_signals: list[TradeSignal] = []
        seen_ids: set[str] = set()

        for strategy in self._strategies:
            if not strategy.is_enabled:
                continue
            signals = strategy.evaluate_batch(markets, context)
            for sig in signals:
                if sig.condition_id not in seen_ids:
                    all_signals.append(sig)
                    seen_ids.add(sig.condition_id)

        # Best signal first (highest EV × confidence)
        return sorted(
            all_signals,
            key=lambda s: s.ev_after_costs * s.confidence,
            reverse=True,
        )

    def get_all_metrics(self) -> list[dict]:
        return [s.get_metrics() for s in self._strategies]

    def __len__(self) -> int:
        return len(self._strategies)
