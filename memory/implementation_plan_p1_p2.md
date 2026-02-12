# Astra V2 — P1/P2 Implementation Plan
**Created:** February 2026
**Objective:** Implement all P1 (critical) and P2 (high-value) improvements from research analysis
**Estimated Total Time:** 3-4 days of focused development

---

## IMPLEMENTATION ORDER (optimized for dependencies)

### Phase 1: Infrastructure & Safety (Day 1 morning)
These are pre-requisites for everything else and add critical safety features.

#### 1. File-Based Kill Switch (30 min) - P1-E
**Why first:** Safety mechanism needed before any other changes. Zero dependencies.

**Files to modify:**
- `main.py` (add kill switch check at top of scan loop)

**Implementation:**
```python
# At top of main.py scan loop
KILL_SWITCH_PATH = Path("/tmp/astra_kill")
if KILL_SWITCH_PATH.exists():
    logger.critical("🛑 KILL SWITCH ACTIVATED - /tmp/astra_kill found. Halting immediately.")
    sys.exit(0)
```

**Test:** `touch /tmp/astra_kill` and verify scanner exits cleanly within 1 scan cycle.

---

#### 2. SQLite Trade Logger (2 hours) - P1-D
**Why second:** Foundation for all analysis, backtesting, and learning. Other features depend on this data.

**Files to create:**
- `scanner/trade_logger.py` — SQLite wrapper with schema
- `scanner/analyze_trades.py` — Analysis utilities (Brier score, calibration plots)

**Database Schema:**
```sql
CREATE TABLE market_snapshots (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    condition_id TEXT NOT NULL,
    question TEXT NOT NULL,
    yes_price REAL NOT NULL,
    no_price REAL NOT NULL,
    liquidity REAL,
    volume_24h REAL,
    end_date TEXT,
    category TEXT
);

CREATE TABLE estimates (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    condition_id TEXT NOT NULL,
    our_estimate REAL NOT NULL,
    confidence REAL NOT NULL,
    robustness_score INTEGER NOT NULL,
    ev_after_costs REAL NOT NULL,
    edge REAL NOT NULL,
    direction TEXT NOT NULL,
    evidence_tier TEXT,
    pro_reasoning TEXT,
    con_reasoning TEXT
);

CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    condition_id TEXT NOT NULL,
    question TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    position_size REAL NOT NULL,
    kelly_pct REAL NOT NULL,
    our_estimate REAL NOT NULL,
    market_price REAL NOT NULL,
    edge REAL NOT NULL,
    pnl REAL,
    pnl_pct REAL,
    outcome TEXT,  -- 'win', 'loss', 'open'
    vix_at_entry REAL,
    category TEXT
);

CREATE TABLE resolutions (
    condition_id TEXT PRIMARY KEY,
    resolved_at TEXT NOT NULL,
    outcome TEXT NOT NULL,  -- 'YES' or 'NO'
    final_yes_price REAL,
    final_no_price REAL
);

CREATE INDEX idx_condition ON market_snapshots(condition_id);
CREATE INDEX idx_timestamp ON market_snapshots(timestamp);
CREATE INDEX idx_trade_condition ON trades(condition_id);
```

**Files to modify:**
- `paper_trader.py` — add logger calls in `open_position()`, `close_position()`, `_update_positions()`
- `main.py` — log market snapshots and estimates during each scan

**Integration points:**
```python
from scanner.trade_logger import TradeLogger

logger_db = TradeLogger("memory/astra_trades.db")

# In main scan loop:
logger_db.log_market_snapshot(market, category)
logger_db.log_estimate(market.condition_id, estimate)

# In paper_trader.py:
logger_db.log_trade_open(position)
logger_db.log_trade_close(position, pnl, outcome)
logger_db.log_resolution(condition_id, outcome, final_price)
```

**Test:** After 1 scan, verify `memory/astra_trades.db` exists and contains records. Run `analyze_trades.py` to compute Brier score on any resolved markets.

---

#### 3. Sharpe/Sortino Tracking (2 hours) - P2-A
**Why third:** Builds on SQLite logger. Enables risk-adjusted performance measurement.

**Files to modify:**
- `paper_trader.py` — add `_calculate_sharpe()` and `_calculate_sortino()` methods
- `paper_trader.py` — call these in `print_status()` if >30 days of history

**Implementation:**
```python
def _calculate_sharpe_ratio(self, lookback_days: int = 30) -> float:
    """Calculate annualized Sharpe ratio from trade history."""
    from scanner.trade_logger import TradeLogger
    logger_db = TradeLogger("memory/astra_trades.db")

    daily_returns = logger_db.get_daily_returns(lookback_days)
    if len(daily_returns) < 7:
        return 0.0

    mean_return = sum(daily_returns) / len(daily_returns)
    std_return = (sum((r - mean_return)**2 for r in daily_returns) / len(daily_returns)) ** 0.5

    if std_return == 0:
        return 0.0

    # Annualize: sqrt(365) scaling factor
    sharpe = (mean_return / std_return) * (365 ** 0.5)
    return sharpe

def _calculate_sortino_ratio(self, lookback_days: int = 30) -> float:
    """Calculate annualized Sortino ratio (downside deviation only)."""
    from scanner.trade_logger import TradeLogger
    logger_db = TradeLogger("memory/astra_trades.db")

    daily_returns = logger_db.get_daily_returns(lookback_days)
    if len(daily_returns) < 7:
        return 0.0

    mean_return = sum(daily_returns) / len(daily_returns)
    downside_returns = [r for r in daily_returns if r < 0]

    if not downside_returns:
        return float('inf')  # No downside = infinite Sortino

    downside_std = (sum(r**2 for r in downside_returns) / len(downside_returns)) ** 0.5

    if downside_std == 0:
        return 0.0

    sortino = (mean_return / downside_std) * (365 ** 0.5)
    return sortino
```

**Add to print_status():**
```python
sharpe = self._calculate_sharpe_ratio(30)
sortino = self._calculate_sortino_ratio(30)

if sharpe != 0:
    table.add_row("📊 Sharpe (30d)", f"{sharpe:.2f}")
    table.add_row("📉 Sortino (30d)", f"{sortino:.2f}")

    if sharpe < 0.5:
        logger.warning("⚠️  Sharpe ratio below 0.5 — strategy may not have statistical edge")
```

**Test:** After 10 closed paper trades, verify Sharpe and Sortino appear in status output.

---

### Phase 2: Core Alpha Signals (Day 1 afternoon)

#### 4. YES+NO Arbitrage Scanner (1 hour) - P1-A
**Why fourth:** Simple, high-value alpha signal. No dependencies.

**Files to modify:**
- `scanner/mispricing_detector.py` — add arbitrage check in `find_opportunities()`

**Implementation:**
```python
# Add after line 102 (after dead-market guard)

        # YES+NO Arbitrage Scanner (P1-A from research)
        # If YES + NO < $1.00, buying both guarantees profit regardless of outcome
        # This is a structural inefficiency, not a probabilistic edge
        arbitrage_edge = 1.0 - (market.yes_price + (1.0 - market.yes_price))
        # Note: NO price = 1 - YES price in Polymarket's binary structure
        # But markets can have YES + NO sum ≠ 1.0 due to fees/liquidity

        # Check if there's a dutch book opportunity
        total_price = market.yes_price + (1.0 - market.yes_price)
        if total_price < 0.98:  # 2% buffer for fees (Polymarket charges ~2% total)
            arbitrage_edge = 1.0 - total_price
            logger.info(
                f"💰 ARBITRAGE OPPORTUNITY: {market.question[:60]}... "
                f"YES={market.yes_price:.3f} + NO={1-market.yes_price:.3f} = {total_price:.3f} "
                f"Edge: {arbitrage_edge:.2%}"
            )
            # Create special arbitrage opportunity (always add, bypasses normal filters)
            opportunities.append(Opportunity(
                market=market,
                estimate=est if est else _create_arbitrage_estimate(market),
                market_price=market.yes_price,
                our_estimate=0.5,  # Doesn't matter for arbitrage
                edge=arbitrage_edge,
                direction="BUY BOTH",  # Special flag for arbitrage
                ev_after_costs=arbitrage_edge - 0.02,  # Minus fees
                robustness_score=5,  # Arbitrage is maximum robustness
                score=arbitrage_edge * 10,  # Boost score heavily
                kelly_pct=0.05,  # Fixed 5% for arbitrage (don't use Kelly here)
            ))
            continue  # Don't process normal edge logic for arbitrage markets
```

**Helper function:**
```python
def _create_arbitrage_estimate(market: Market) -> Estimate:
    """Create a dummy Estimate for arbitrage opportunities (no LLM needed)."""
    from scanner.probability_estimator import Estimate
    return Estimate(
        market_condition_id=market.condition_id,
        probability=0.5,
        confidence=1.0,
        reasoning="Arbitrage opportunity - no prediction needed",
        evidence_tier="A",
        robustness_score=5,
        trap_flags=[],
        no_trade=False,
        ev_after_costs=0.0,
        kelly_position_pct=0.05
    )
```

**Test:** Manually check current markets for any with YES+NO < 0.98. If found, verify scanner flags them.

---

#### 5. VIX Regime Labels in Prompts (1 hour) - P2-B
**Why fifth:** Astra already fetches VIX. Simple enhancement with high impact on macro-sensitive markets.

**Files to modify:**
- `scanner/probability_estimator.py` — add VIX regime to PRO/CON/Synthesizer prompts

**Implementation:**
```python
# Add helper function at top of probability_estimator.py
def _get_vix_regime_label(vix: float) -> str:
    """Map VIX level to regime label for Claude context."""
    if vix < 12:
        return "VERY LOW (<12) — Extreme complacency"
    elif vix < 16:
        return "LOW (12-16) — Market calm"
    elif vix < 20:
        return "MEDIUM (16-20) — Normal volatility"
    elif vix < 30:
        return "ELEVATED (20-30) — Uncertainty rising"
    elif vix < 40:
        return "HIGH (30-40) — Fear/stress regime"
    else:
        return "EXTREME (>40) — Panic/crisis regime"

# In _estimate_batch_adversarial(), before building PRO_PROMPT:
vix_regime = _get_vix_regime_label(vix_level) if vix_level else "UNKNOWN"

# Add to PRO_PROMPT and CON_PROMPT (before the market list):
f"""
Current market environment:
- VIX (volatility index): {vix_level:.1f} — Regime: {vix_regime}
- This affects risk appetite and probability assessment for macro/political events
"""

# Add to SYNTHESIZER_PROMPT:
f"""
Market context: VIX is {vix_level:.1f} ({vix_regime})
When VIX is HIGH/EXTREME, increase confidence penalties for overconfident predictions.
When VIX is LOW, markets may be underpricing tail risks.
"""
```

**Test:** Run a scan and check logs to verify VIX regime appears in prompt context. Compare estimates before/after on same markets.

---

### Phase 3: Architecture & Documentation (Day 2 morning)

#### 6. BaseStrategy Architecture Design (1 hour) - P1-C
**Why sixth:** Architectural foundation, but doesn't change current behavior. Prepare for future extensibility.

**Files to create:**
- `scanner/base_strategy.py` — abstract base class
- `docs/base_strategy_design.md` — architecture documentation

**Implementation:**
```python
# scanner/base_strategy.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from scanner.market_fetcher import Market

@dataclass
class StrategySignal:
    """Output from a strategy's analysis."""
    condition_id: str
    direction: str  # "BUY YES", "BUY NO", "BUY BOTH", "HOLD"
    edge: float
    confidence: float
    reasoning: str
    signal_type: str  # "arbitrage", "llm_estimate", "statistical", etc.
    kelly_pct: float

class BaseStrategy(ABC):
    """
    Abstract base class for all Astra trading strategies.

    Based on Novus-Tech/Polymarket-Arbitrage-Bot BaseStrategy pattern.
    Each strategy implements:
    - on_market_update(): process new market data
    - on_book_update(): process orderbook changes (Phase 3)
    - on_tick(): periodic evaluation (e.g., every 10 seconds)
    """

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    @abstractmethod
    def on_market_update(self, market: Market) -> Optional[StrategySignal]:
        """
        Called when market data is refreshed (every scan cycle).

        Returns StrategySignal if opportunity found, else None.
        """
        pass

    @abstractmethod
    def on_book_update(self, condition_id: str, orderbook: dict) -> Optional[StrategySignal]:
        """
        Called when Level-2 orderbook data changes (Phase 3 only).

        For Phase 1-2, this can return None.
        """
        pass

    @abstractmethod
    def on_tick(self) -> list[StrategySignal]:
        """
        Called periodically (e.g., every 10 seconds) for time-based logic.

        Returns list of signals from all monitored markets.
        """
        pass

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
```

**Example strategy implementations:**
```python
# scanner/strategies/arbitrage_strategy.py
from scanner.base_strategy import BaseStrategy, StrategySignal

class ArbitrageStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("YES+NO Arbitrage")

    def on_market_update(self, market: Market) -> Optional[StrategySignal]:
        total_price = market.yes_price + (1.0 - market.yes_price)
        if total_price < 0.98:
            edge = 1.0 - total_price
            return StrategySignal(
                condition_id=market.condition_id,
                direction="BUY BOTH",
                edge=edge,
                confidence=1.0,
                reasoning=f"Dutch book: YES+NO = {total_price:.3f} < 1.0",
                signal_type="arbitrage",
                kelly_pct=0.05
            )
        return None

    def on_book_update(self, condition_id: str, orderbook: dict):
        return None  # Not used in Phase 2

    def on_tick(self):
        return []  # Arbitrage is evaluated on market updates only

# scanner/strategies/llm_estimate_strategy.py
class LLMEstimateStrategy(BaseStrategy):
    def __init__(self, estimator):
        super().__init__("Claude Adversarial Estimation")
        self.estimator = estimator

    def on_market_update(self, market: Market) -> Optional[StrategySignal]:
        # Calls probability_estimator.py logic
        estimate = self.estimator.estimate_single(market)
        if estimate and not estimate.no_trade:
            edge = estimate.probability - market.yes_price
            if abs(edge) >= MISPRICING_THRESHOLD:
                return StrategySignal(
                    condition_id=market.condition_id,
                    direction="BUY YES" if edge > 0 else "BUY NO",
                    edge=edge,
                    confidence=estimate.confidence,
                    reasoning=estimate.reasoning,
                    signal_type="llm_estimate",
                    kelly_pct=estimate.kelly_position_pct
                )
        return None

    # ... etc
```

**Migration plan:**
1. Create base classes (this step)
2. Refactor `mispricing_detector.py` to use strategies (next sprint)
3. Add new strategies as plugins without touching core code

**Test:** Create `ArbitrageStrategy` and verify it produces same signals as current inline code.

---

#### 7. CLOB Credential Refresh Documentation (30 min) - P1-B
**Why seventh:** Critical for Phase 3 but doesn't require code now. Document the requirement.

**Files to create:**
- `docs/phase3_clob_requirements.md`

**Content:**
```markdown
# Phase 3: Live Trading CLOB Requirements

## Critical: Hourly Credential Refresh

**Source:** Research analysis P1-B (github.com/Lazydayz137/Polymarket-spike-bot-v1)

**Problem:** Polymarket's `py-clob-client` API keys expire every **60 minutes**. Without refresh, all order placement calls silently fail after 1 hour of runtime.

**Solution:** Implement automatic credential refresh loop.

### Implementation Pattern

```python
from py_clob_client.client import ClobClient
from datetime import datetime, timedelta
import threading

class PolymarketClient:
    def __init__(self, key, secret, passphrase):
        self.key = key
        self.secret = secret
        self.passphrase = passphrase
        self.client = None
        self.last_refresh = None
        self.refresh_interval = timedelta(minutes=55)  # Refresh 5 min before expiry

        self._refresh_credentials()
        self._start_refresh_thread()

    def _refresh_credentials(self):
        """Re-initialize CLOB client with fresh credentials."""
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=self.key,
            secret=self.secret,
            passphrase=self.passphrase
        )
        self.last_refresh = datetime.now()
        logger.info(f"🔄 CLOB credentials refreshed at {self.last_refresh}")

    def _start_refresh_thread(self):
        """Background thread that refreshes credentials every 55 minutes."""
        def refresh_loop():
            while True:
                time.sleep(self.refresh_interval.total_seconds())
                self._refresh_credentials()

        thread = threading.Thread(target=refresh_loop, daemon=True)
        thread.start()

    def place_order(self, *args, **kwargs):
        """Wrapper that ensures credentials are fresh before order placement."""
        if datetime.now() - self.last_refresh > self.refresh_interval:
            self._refresh_credentials()

        return self.client.place_order(*args, **kwargs)
```

### Testing

Before live deployment:
1. Create test client
2. Wait 61 minutes
3. Attempt order placement
4. Verify auto-refresh occurred and order succeeded

### Contract Approvals Required

Before any CLOB trading, the wallet must approve three contracts on Polygon:
1. **CTF Exchange** - Conditional token framework
2. **Neg Risk Exchange** - Negative risk markets
3. **Conditional Tokens** - ERC-1155 token contract

See `runesatsdev/polymarket-trading-bot` repo scripts/ for approval workflow.

## Additional Phase 3 Requirements

- [ ] Polygon RPC endpoint (Alchemy/Infura)
- [ ] Wallet with USDC on Polygon
- [ ] Private key secure storage (not in .env — use encrypted keystore)
- [ ] Gas price monitoring (set max gas to prevent rugs)
- [ ] Order timeout handling (cancel unfilled orders after 10s)
- [ ] Fill confirmation (don't assume order filled until confirmed)
- [ ] WebSocket orderbook streaming (upgrade from REST polling)
```

**Test:** Review document with Phase 3 checklist. No code to test yet.

---

#### 8. Audit + Memory Agent Architecture Doc (1 hour) - P1-F
**Why eighth:** Architectural documentation for future agent roles. Informs how SQLite logger feeds learning.

**Files to create:**
- `docs/agent_architecture.md`

**Content:**
```markdown
# Astra V2: Multi-Agent Architecture

**Source:** arXiv:2512.02227 "Orchestration Framework for Financial Agents"

## Current Agents (Phase 2)

1. **PRO Agent** - Generates bullish probability estimate with supporting evidence
2. **CON Agent** - Generates bearish probability estimate with contradictory evidence
3. **Synthesizer Agent** - Resolves PRO/CON disagreement into final calibrated estimate
4. **Learning Agent** - Analyzes resolved trades and proposes strategy overrides

## Proposed Agent Additions (Phase 3)

### 5. Audit Agent
**Role:** Log every estimate with full decision metadata for post-hoc analysis.

**Responsibilities:**
- Record PRO/CON divergence magnitude (disagreement = signal uncertainty)
- Log evidence tier breakdown (how much A/B/C/D tier data supported estimate)
- Track which markets triggered trap flags
- Flag estimates with low robustness scores for review
- Compute daily/weekly Brier scores by market category

**Implementation:** Extend SQLite `estimates` table with:
```sql
ALTER TABLE estimates ADD COLUMN pro_estimate REAL;
ALTER TABLE estimates ADD COLUMN con_estimate REAL;
ALTER TABLE estimates ADD COLUMN divergence REAL;  -- abs(pro - con)
ALTER TABLE estimates ADD COLUMN trap_flags TEXT;  -- JSON array
```

**Output:** Weekly calibration report comparing estimated probabilities vs resolution outcomes.

### 6. Memory Agent
**Role:** Aggregate long-term patterns to improve future estimation.

**Responsibilities:**
- Identify systematic biases (e.g., "Astra overestimates YES on crypto markets by 8%")
- Track category-specific calibration (sports vs politics vs crypto)
- Detect regime changes (e.g., "calibration degraded after VIX >30 event")
- Propose calibration corrections to Learning Agent

**Implementation:** Analyze `trades` + `resolutions` tables:
```python
def analyze_category_bias(category: str, lookback_days: int = 90):
    """Compute mean error for a market category."""
    trades = db.get_resolved_trades(category, lookback_days)
    errors = [trade.our_estimate - trade.actual_outcome for trade in trades]
    mean_bias = sum(errors) / len(errors)

    if abs(mean_bias) > 0.05:
        return {
            "category": category,
            "bias": mean_bias,
            "correction": -mean_bias,  # Offset in opposite direction
            "confidence": len(trades) / 30  # More data = higher confidence
        }
```

**Output:** Calibration corrections fed to `strategy_overrides.json`:
```json
{
    "CATEGORY_CALIBRATION": {
        "crypto": -0.08,
        "sports": +0.03,
        "politics": -0.01
    },
    "expires": "2026-04-01"
}
```

### 7. Risk Agent (Phase 4)
**Role:** Portfolio-level risk management across multiple simultaneous positions.

**Responsibilities:**
- Enforce correlation limits (don't open 5 positions on correlated crypto markets)
- Implement drawdown-based position sizing (reduce Kelly fraction after losses)
- Detect concentration risk (too much capital in one market category)
- Dynamic VaR calculation and alerts

### 8. Execution Agent (Phase 4)
**Role:** Optimal order placement on Polymarket CLOB.

**Responsibilities:**
- Smart order routing (maker vs taker based on urgency)
- Order splitting (TWAP for large positions to minimize slippage)
- Fill monitoring and partial fill handling
- Gas price optimization

## Agent Interaction Flow (Phase 3)

```
Market Data → PRO + CON (parallel) → Synthesizer → Audit Agent → Risk Agent → Execution Agent
                                                         ↓
                                                    SQLite Logger
                                                         ↓
                                                    Memory Agent → Learning Agent → strategy_overrides.json
```

## Testing Strategy

Each agent should have:
1. **Unit tests** - Mock inputs, verify outputs
2. **Integration tests** - Agent interactions with SQLite logger
3. **Backtest validation** - Run agent on historical data, measure improvement

## Migration Path

**Phase 2 (current):**
- ✅ PRO, CON, Synthesizer, Learning agents active
- ⏳ Add Audit Agent logging (P1-F)
- ⏳ Add Memory Agent bias detection (P1-F)

**Phase 3 (next sprint):**
- Add Risk Agent for portfolio-level controls
- Add Execution Agent for CLOB order placement

**Phase 4 (future):**
- Add Research Agent for market discovery
- Add Calibration Agent for real-time estimate adjustment
```

**Test:** Review document. No code to test yet.

---

### Phase 4: User Experience (Day 2 afternoon)

#### 9. Telegram Notifier (3 hours) - P2-C
**Why ninth:** Remote monitoring. Requires setup but high operational value.

**Files to create:**
- `scanner/telegram_notifier.py`
- `docs/telegram_setup.md`

**Implementation:**
```python
# scanner/telegram_notifier.py
import os
from telegram import Bot
from telegram.error import TelegramError
import logging

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """
    Send trading alerts to Telegram.

    Setup:
    1. Create bot via @BotFather on Telegram
    2. Get bot token
    3. Send /start to your bot
    4. Get your chat_id from @userinfobot
    5. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
    """

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)

        if self.enabled:
            self.bot = Bot(token=self.token)
            logger.info("✅ Telegram notifier initialized")
        else:
            logger.warning("⚠️  Telegram notifier disabled (missing credentials)")

    def notify(self, message: str, silent: bool = False):
        """Send message to Telegram. silent=True for low-priority alerts."""
        if not self.enabled:
            return

        try:
            self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="Markdown",
                disable_notification=silent
            )
        except TelegramError as e:
            logger.error(f"Telegram send failed: {e}")

    def notify_opportunity(self, opportunity):
        """Alert on high-score opportunity."""
        if opportunity.score < 0.15:
            return  # Only alert on strong signals

        msg = f"""
🎯 *OPPORTUNITY DETECTED*

Market: {opportunity.market.question[:80]}
Direction: {opportunity.direction}
Edge: {opportunity.edge:+.1%}
Score: {opportunity.score:.3f}
Kelly: {opportunity.kelly_pct:.1%}

Confidence: {opportunity.estimate.confidence:.0%}
Robustness: {opportunity.robustness_score}/5
EV: {opportunity.ev_after_costs:+.2%}
        """
        self.notify(msg)

    def notify_position_opened(self, position):
        """Alert when paper position opened."""
        msg = f"""
📈 *POSITION OPENED*

{position.question[:80]}
{position.direction} @ {position.entry_price:.3f}
Size: ${position.position_size:.2f} ({position.kelly_pct:.1%} Kelly)
Edge: {position.edge:+.1%}
        """
        self.notify(msg, silent=True)

    def notify_position_closed(self, position, pnl: float, outcome: str):
        """Alert when paper position closed."""
        emoji = "✅" if pnl > 0 else "❌"
        msg = f"""
{emoji} *POSITION CLOSED*

{position.question[:80]}
Outcome: {outcome}
P&L: ${pnl:+.2f} ({pnl/position.position_size:+.1%})
Held: {(position.closed_at - position.opened_at).days}d
        """
        self.notify(msg)

    def notify_daily_summary(self, stats: dict):
        """Send end-of-day summary."""
        msg = f"""
📊 *DAILY SUMMARY*

Open Positions: {stats['open_count']}
Total Exposure: ${stats['total_exposure']:.2f}
Unrealized P&L: ${stats['unrealized_pnl']:+.2f}

Closed Today: {stats['closed_today']}
Realized P&L: ${stats['realized_pnl_today']:+.2f}

Bankroll: ${stats['current_bankroll']:.2f}
        """
        self.notify(msg, silent=True)

    def notify_alert(self, alert_type: str, message: str):
        """Critical system alerts."""
        emoji_map = {
            "error": "🔴",
            "warning": "⚠️",
            "kill_switch": "🛑",
            "api_limit": "💰"
        }
        emoji = emoji_map.get(alert_type, "ℹ️")
        self.notify(f"{emoji} *{alert_type.upper()}*\n\n{message}")
```

**Integration in main.py:**
```python
from scanner.telegram_notifier import TelegramNotifier

notifier = TelegramNotifier()

# After finding opportunities:
for opp in opportunities[:3]:  # Top 3 only
    notifier.notify_opportunity(opp)

# In paper_trader.py:
notifier.notify_position_opened(position)
notifier.notify_position_closed(position, pnl, outcome)
```

**Setup guide:**
```markdown
# docs/telegram_setup.md

## Telegram Bot Setup (5 minutes)

1. Open Telegram, search for @BotFather
2. Send `/newbot`
3. Follow prompts to name your bot (e.g., "Astra Trading Alerts")
4. Copy the token (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)
5. Search for @userinfobot, send `/start`
6. Copy your chat_id (looks like `987654321`)
7. Add to `.env`:
```
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```
8. Send `/start` to your new bot
9. Run Astra — you should get a "Telegram notifier initialized" message
10. Trigger an opportunity to test

## Privacy Note

- Bot token is public-facing (used to send messages TO you)
- Chat ID is your private identifier
- Never share your `.env` file
```

**Test:** Set up bot, trigger notification via `notifier.notify("Test message")`.

---

## IMPLEMENTATION CHECKLIST

```
Day 1 Morning (Infrastructure & Safety):
[ ] 1. File-based kill switch (30 min)
[ ] 2. SQLite trade logger with schema (2 hours)
[ ] 3. Sharpe/Sortino tracking (2 hours)

Day 1 Afternoon (Core Alpha Signals):
[ ] 4. YES+NO arbitrage scanner (1 hour)
[ ] 5. VIX regime labels in prompts (1 hour)

Day 2 Morning (Architecture & Documentation):
[ ] 6. BaseStrategy architecture design (1 hour)
[ ] 7. CLOB credential refresh docs (30 min)
[ ] 8. Audit + Memory agent architecture docs (1 hour)

Day 2 Afternoon (User Experience):
[ ] 9. Telegram notifier (3 hours)

Total: ~12 hours (1.5 days focused work)
```

## TESTING STRATEGY

After each implementation:
1. **Syntax check:** `python3.12 -c "import <module>"`
2. **Unit test:** Test function in isolation
3. **Integration test:** Run full scan with new feature enabled
4. **Regression test:** Verify existing features still work

After all P1 items:
1. **End-to-end test:** Run paper trader for 24 hours with all features enabled
2. **Performance test:** Verify scan time <30s with SQLite logging
3. **Failure test:** Trigger kill switch, verify clean shutdown
4. **Alert test:** Verify Telegram notifications on all event types

## ROLLBACK PLAN

Each feature is independently toggleable:
- SQLite logger: set `ENABLE_TRADE_LOGGER=false` in .env
- Telegram: leave `TELEGRAM_BOT_TOKEN` empty
- Arbitrage scanner: comment out check in mispricing_detector.py
- VIX regime: prompt templates backward-compatible

Git commit after each completed item for easy rollback.
