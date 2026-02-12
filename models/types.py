"""
Core data models for allocator-grade Polymarket system.
All dataclasses follow ChatGPT-approved Core Spec v1.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# MARKET STATE MACHINE (CRITICAL FIX #3)
# ============================================================================

class MarketState(str, Enum):
    """Market lifecycle states per three-clock time model."""
    NORMAL = "normal"
    WATCH = "watch"
    CLOSE_WINDOW = "close_window"
    POST_CLOSE = "post_close"
    PROPOSED = "proposed"
    CHALLENGE_WINDOW = "challenge_window"
    RESOLVED = "resolved"


# ============================================================================
# EVENT & MARKET (with negRisk flags, CRITICAL FIX #17)
# ============================================================================

@dataclass
class Event:
    """
    Polymarket event (can contain multiple markets).
    CRITICAL: negRisk and augmentedNegRisk flags affect cluster assignment.
    """
    event_id: str
    title: str
    neg_risk: bool = False  # negRisk events break parity assumptions
    augmented_neg_risk: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class Market:
    """
    Market with three-clock time model and execution-critical fields.

    CRITICAL FIXES:
    - #3: time_to_close (NOT hours_to_expiry)
    - #2: feeRateBps from metadata (NOT hardcoded)
    - #17: event with negRisk flags
    """
    condition_id: str
    question: str
    description: str

    # Token IDs
    yes_token_id: str
    no_token_id: str

    # Prices (from order books)
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float

    # Three-clock time model (CRITICAL FIX #3)
    time_to_close: Optional[float] = None  # Hours until trading ends
    time_to_proposal_expected: Optional[float] = None  # Hours to resolution proposal
    challenge_window_start: Optional[int] = None  # Unix timestamp when challenge period begins

    # Market metadata
    category: str = "other"
    liquidity: float = 0.0
    volume_24h: float = 0.0
    tick_size: float = 0.01  # Usually 0.01

    # Fee regime (CRITICAL FIX #2)
    fee_rate_bps: int = 200  # Basis points (e.g., 200 = 2%)

    # Resolution metadata
    rules_text: str = ""
    resolution_source: str = ""

    # State (computed)
    state: MarketState = MarketState.NORMAL

    # Event association (CRITICAL FIX #17)
    event: Optional[Event] = None

    # Metadata from API
    raw_metadata: dict = field(default_factory=dict)

    @property
    def mid_price(self) -> Optional[float]:
        """
        Convenience: mid price from YES book.

        DEPRECATED: Use execution.mid.compute_mid() for production (handles fallbacks correctly).
        This property is kept for backward compatibility only.
        """
        from execution.mid import compute_mid
        return compute_mid(self.yes_bid, self.yes_ask)


# ============================================================================
# ORDER BOOK (with staleness tracking)
# ============================================================================

@dataclass
class OrderBook:
    """
    L2 order book for a token (YES or NO).

    Includes staleness and churn tracking for QS computation.
    """
    token_id: str

    # Best prices
    best_bid: float
    best_ask: float

    # Depth at levels (for QS depth scoring)
    bids: list[tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    asks: list[tuple[float, float]] = field(default_factory=list)

    # Staleness tracking
    timestamp_ms: int = 0  # Unix timestamp in milliseconds
    timestamp_age_ms: int = 0  # Age since last update

    # Churn tracking (for QS churn scoring)
    churn_rate: float = 0.0  # Updates per second (rolling)

    # Last known mid (for fallback)
    last_mid: Optional[float] = None


# ============================================================================
# FILL (with maker/taker flag for markout tracking)
# ============================================================================

@dataclass
class Fill:
    """
    Order fill event from user WebSocket.

    CRITICAL:
    - Includes maker/taker flag for Truth Report separation
    - GAP #2 FIX: Enhanced classification logic
    """
    fill_id: str
    order_id: str
    condition_id: str
    token_id: str

    side: Literal["BUY", "SELL"]
    price: float
    size_tokens: float

    # Execution metadata
    timestamp: int  # Unix timestamp (milliseconds)
    maker: bool = True  # True if maker fill, False if taker

    # Fee charged
    fee_rate_bps: int = 200
    fee_paid_usd: float = 0.0

    # For markout tracking
    mid_at_fill: Optional[float] = None

    # GAP #2 FIX: Confidence in maker/taker classification
    classification_source: Literal["WS_FLAG", "POST_ONLY", "SPREAD_CROSS", "UNKNOWN"] = "UNKNOWN"

    def classify_maker_taker(self, order: "StoredOrder", book_mid: Optional[float] = None) -> bool:
        """
        Robust maker vs taker classification (GAP #2 FIX).

        CRITICAL: Don't trust WS flag alone.

        Truth sources (in priority order):
        1. Order origin (post_only maker orders)
        2. Spread crossing detection (if we crossed spread → taker)
        3. WS maker flag (if present and trusted)

        Args:
            order: StoredOrder that generated this fill
            book_mid: Mid price at time of fill (for spread crossing detection)

        Returns:
            True if maker fill, False if taker
        """
        # Priority 1: post_only orders are always maker
        if order.post_only and order.origin == "MAKER_QUOTE":
            self.classification_source = "POST_ONLY"
            return True

        # Priority 2: Spread crossing detection
        if book_mid is not None:
            spread_crossed = False
            if order.side == "BUY" and self.price > book_mid:
                spread_crossed = True  # Bought above mid → aggressive taker
            elif order.side == "SELL" and self.price < book_mid:
                spread_crossed = True  # Sold below mid → aggressive taker

            if spread_crossed:
                self.classification_source = "SPREAD_CROSS"
                return False

        # Priority 3: WS flag (if available)
        if self.maker is not None:
            self.classification_source = "WS_FLAG"
            return self.maker

        # Fallback: Assume maker if uncertain (conservative)
        self.classification_source = "UNKNOWN"
        return True


# ============================================================================
# ORDER INTENT (internal representation)
# ============================================================================

@dataclass
class OrderIntent:
    """
    Internal order specification before submission.

    Uses human-friendly fields. Converted to SignedOrderPayload for submission.

    GAP #2 FIX: Includes origin for maker/taker classification.
    """
    condition_id: str
    token_id: str

    side: Literal["BUY", "SELL"]
    price: float  # Human-readable price (0-1)
    size_in_tokens: float

    order_type: Literal["GTC", "GTD", "FOK", "FAK"] = "GTD"
    post_only: bool = True  # CRITICAL: Only valid with GTC/GTD

    # GTD expiration (CRITICAL FIX #1, #9, #20)
    expiration: Optional[int] = None  # Unix timestamp (required for GTD)

    # Fee regime
    fee_rate_bps: int = 200

    # GAP #2 FIX: Order origin for maker/taker classification
    origin: Literal["MAKER_QUOTE", "TAKER_ARB", "SATELLITE", "OTHER"] = "MAKER_QUOTE"


# ============================================================================
# SIGNED ORDER PAYLOAD (execution-truth fields, CRITICAL FIX #5)
# ============================================================================

@dataclass
class SignedOrderPayload:
    """
    Signed order payload for CLOB submission.

    CRITICAL FIX #5: Includes execution-truth fields for reconciliation.
    """
    # Human-friendly fields
    token_id: str
    price: float
    size: float
    side: Literal["BUY", "SELL"]

    # Order type fields (CRITICAL FIX #1)
    order_type: Literal["GTC", "GTD", "FOK", "FAK"]
    expiration: Optional[int] = None  # Unix timestamp for GTD

    # Execution-truth fields for reconciliation
    maker_amount: str = ""  # Actual signed amount
    taker_amount: str = ""  # Actual signed amount
    nonce: int = 0
    salt: int = 0
    signature: str = ""

    # Fee
    fee_rate_bps: int = 200


# ============================================================================
# STORED ORDER (Order State Store, CRITICAL FIX #5)
# ============================================================================

class OrderStatus(str, Enum):
    """Order lifecycle states."""
    PENDING = "pending"
    LIVE = "live"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"


@dataclass
class StoredOrder:
    """
    Persisted order in Order State Store.

    CRITICAL FIXES:
    - #5: Stores BOTH human-friendly AND execution-truth fields
    - GAP #2: Order origin tracking for maker/taker classification
    - GAP #3: Partial fill tracking
    """
    # Identifier
    order_id: str

    # Human-friendly fields
    condition_id: str
    token_id: str
    side: Literal["BUY", "SELL"]
    price: float
    size_in_tokens: float

    order_type: Literal["GTC", "GTD", "FOK", "FAK"]
    post_only: bool
    expiration: Optional[int] = None  # Unix timestamp
    fee_rate_bps: int = 200

    # Execution-truth fields (for reconciliation)
    maker_amount: str = ""
    taker_amount: str = ""
    nonce: int = 0
    salt: int = 0
    signature: str = ""
    order_hash: str = ""  # CLOB order hash

    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    placed_at: str = ""  # ISO timestamp
    last_seen_ws: str = ""  # ISO timestamp

    # GAP #2 FIX: Order origin for maker/taker classification
    origin: Literal["MAKER_QUOTE", "TAKER_ARB", "SATELLITE", "OTHER"] = "OTHER"

    # GAP #3 FIX: Partial fill tracking
    original_size: float = 0.0  # Initial size at submission
    remaining_size: float = 0.0  # After partial fills
    filled_size: float = 0.0  # Cumulative filled

    # Cancellation metadata
    cancel_reason: Optional[str] = None
    clob_error: Optional[str] = None


# ============================================================================
# CLUSTER ASSIGNMENT
# ============================================================================

@dataclass
class Cluster:
    """
    Risk cluster for correlated markets.

    CRITICAL: Deterministic assignment (same market → same cluster_id).
    """
    cluster_id: str
    markets: list[str] = field(default_factory=list)  # condition_ids

    # Determinism metadata (for reproducibility)
    entities: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    resolution_source: str = ""

    # negRisk handling (CRITICAL FIX #17)
    neg_risk_event_id: Optional[str] = None  # If set, all markets in event → one cluster


# ============================================================================
# PORTFOLIO EXPOSURE (CRITICAL FIX #5, #18)
# ============================================================================

@dataclass
class PortfolioExposure:
    """
    Current portfolio exposure state.

    CRITICAL FIXES:
    - #5: Token-level inventory, mark-to-mid USD
    - #18: Balance reservations from open orders
    """
    # Per-market exposure (mark-to-mid USD)
    market_exposures: dict[str, float] = field(default_factory=dict)  # condition_id → USD

    # Per-cluster exposure
    cluster_exposures: dict[str, float] = field(default_factory=dict)  # cluster_id → USD

    # Token-level inventory (CRITICAL FIX #5)
    token_inventory: dict[str, float] = field(default_factory=dict)  # token_id → token count

    # Balance reservations (CRITICAL FIX #18)
    reserved_usdc_by_market: dict[str, float] = field(default_factory=dict)  # condition_id → USDC
    reserved_tokens_by_token_id: dict[str, float] = field(default_factory=dict)  # token_id → tokens

    # Aggregate
    total_exposure_usd: float = 0.0

    # Risk budget tracking
    satellite_risk_used_usd: float = 0.0
    taker_risk_used_usd: float = 0.0


# ============================================================================
# ARBITRAGE OPPORTUNITY (CRITICAL FIX #7)
# ============================================================================

@dataclass
class ArbitrageOpp:
    """
    Structural arbitrage opportunity.

    CRITICAL FIX #7: Execution mode + leg risk awareness.
    """
    type: Literal["YES_NO_PARITY", "DUPLICATE_MARKETS"]

    markets: list[Market]
    legs: list[dict]  # [{"token": "YES", "side": "BUY", "price": 0.52}, ...]

    expected_profit: float  # After fees
    execution_mode: Literal["maker", "taker"] = "taker"

    # Leg risk metadata (for taker mode)
    max_leg_time_ms: int = 5000  # 5s default
    requires_atomic: bool = True  # Unwind if leg 2 fails
