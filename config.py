# Core Spec v1 Configuration
# Allocator-Grade Polymarket Trading System
# ChatGPT-Approved: All 21 fixes applied

from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# ============================================================================
# CREDENTIALS & WALLET
# ============================================================================

POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY")
POLY_API_KEY = os.getenv("POLY_API_KEY")
POLY_API_SECRET = os.getenv("POLY_API_SECRET")
POLY_API_PASSPHRASE = os.getenv("POLY_API_PASSPHRASE")

# External API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Claude model configuration
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Wallet configuration
TRADING_ADDRESS = os.getenv("TRADING_ADDRESS")  # Your Polygon address
SIGNATURE_TYPE = int(os.getenv("SIGNATURE_TYPE", "0"))  # 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE
FUNDER_ADDRESS = os.getenv("FUNDER_ADDRESS")  # Funder address for signature type

# ============================================================================
# TRADING MODE
# ============================================================================

# PAPER_MODE: If True, simulate fills using PaperTradingSimulator (GAP #8)
#             If False, use live WebSocket fills
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() in ("true", "1", "yes")

# ============================================================================
# POLYMARKET API ENDPOINTS
# ============================================================================

CLOB_API_URL = "https://clob.polymarket.com"
GAMMA_API_URL = "https://gamma-api.polymarket.com"

# ============================================================================
# CLOB EXECUTION LAYER
# ============================================================================

# Batch limits (CRITICAL FIX #4)
MAX_BATCH_ORDERS = 15  # Polymarket CLOB hard limit

# GTD expiration (CRITICAL FIX #1, #9, #20)
GTD_SAFETY_BUFFER_SECONDS = 60  # Configurable safety threshold
GTD_DESIRED_SECONDS = 120  # How long we want orders to live
# Total GTD expiration = now + GTD_SAFETY_BUFFER_SECONDS + GTD_DESIRED_SECONDS

# Order type validation (CRITICAL FIX #1)
VALID_ORDER_TYPES = ["GTC", "GTD", "FOK", "FAK"]
POST_ONLY_ALLOWED_TYPES = ["GTC", "GTD"]  # postOnly only valid with these

# WebSocket staleness
WS_STALENESS_THRESHOLD_MS = 5000  # Cancel-all if no WS update for 5s
WS_RECONNECT_DELAY_SECONDS = 5

# Rate limiting (CRITICAL FIX #14)
MUTATION_MAX_PER_CYCLE = 15  # Max cancel/replace per cycle
MUTATION_MAX_PER_MINUTE = 60  # Rolling budget
MUTATION_MIN_DRIFT_TICKS = 2  # Only mutate if drift > 2 ticks
MAX_REQUESTS_PER_MINUTE = 80  # General API rate limit

# Reconciliation
RECONCILE_ON_STARTUP = True
RECONCILE_INTERVAL_SECONDS = 300  # Every 5 minutes

# ============================================================================
# PORTFOLIO RISK ENGINE (CRITICAL FIX #5)
# ============================================================================

# Capital
BANKROLL = 5000.0  # Initial bankroll in USD

# Cluster caps (CRITICAL FIX: tightened from 20% to 12%)
MAX_CLUSTER_EXPOSURE_PCT = 0.12  # 12% of bankroll per cluster
MAX_AGG_EXPOSURE_PCT = 0.40  # 40% total at-risk capital
MAX_MARKET_INVENTORY_PCT = 0.01  # 1% per market max

# Satellite risk budget
SATELLITE_RISK_BUDGET_PCT = 0.15  # 15% for info trades (Astra V2)

# Taker risk budget (CRITICAL FIX #7)
TAKER_RISK_BUDGET_PCT = 0.05  # 5% max for taker trades (parity arb)
MAX_LEG_TIME_MS = 5000  # 5 seconds max between parity legs

# Near-resolution caps (CRITICAL FIX #21)
NEAR_RESOLUTION_HOURS = 48  # Tighten caps when < 48h to close
NEAR_RESOLUTION_CAP_MULTIPLIER = 0.5  # Halve position limits

# ============================================================================
# MARKET TIME MODEL & STATE MACHINE (CRITICAL FIX #3)
# ============================================================================

# State thresholds (in hours to close)
STATE_WATCH_THRESHOLD_HOURS = 72  # NORMAL → WATCH
STATE_CLOSE_WINDOW_THRESHOLD_HOURS = 24  # WATCH → CLOSE_WINDOW

# Challenge window
CHALLENGE_WINDOW_DURATION_HOURS = 2  # 2 hours post-proposal

# ============================================================================
# RESOLUTION RISK SCORE (RRS)
# ============================================================================

# Hard veto gates
RRS_VETO_MAKER = 0.35  # No maker quoting if RRS > 0.35
RRS_VETO_SATELLITE = 0.25  # No satellite trades if RRS > 0.25
RRS_NEAR_RESOLUTION_TIGHTEN = 0.20  # Stricter near resolution

# Component weights
RRS_WEIGHT_AMBIGUITY = 0.40
RRS_WEIGHT_DISPUTE = 0.30
RRS_WEIGHT_TIME = 0.20
RRS_WEIGHT_CLARIFICATION = 0.10

# Dispute priors by category
DISPUTE_PRIORS = {
    "crypto": 0.15,
    "sports": 0.20,
    "politics": 0.30,
    "other": 0.25,
}

# ============================================================================
# QUOTEABILITY SCORE (QS) (CRITICAL FIX #4, #12)
# ============================================================================

# Active set
ACTIVE_QUOTE_COUNT = 40  # Top N markets to actively quote
MAX_MARKETS_PER_CLUSTER_IN_ACTIVE_SET = 5

# Scoring weights
QS_WEIGHT_DEPTH = 0.35
QS_WEIGHT_CHURN = 0.25
QS_WEIGHT_JUMP = 0.20
QS_WEIGHT_RESOLUTION = 0.20

# Vetoes
QS_MIN_LIQUIDITY = 500.0  # $500 min liquidity
QS_RECENT_JUMP_THRESHOLD = 0.05  # 5% jump in last hour

# Staleness thresholds (mode-aware for paper trading compatibility)
QS_BOOK_STALENESS_S_PROD = 5  # 5s max book age in production
QS_BOOK_STALENESS_S_PAPER = 120  # 120s max book age in paper mode (2x cycle time)

# ============================================================================
# MARKET-MAKER QUOTING POLICY
# ============================================================================

# Fair value band
FV_BASE_HALF_WIDTH = 0.015  # 1.5¢ base half-width
FV_HIGH_CHURN_MULTIPLIER = 1.5  # Widen for churn > 0.5
FV_JUMP_RISK_MULTIPLIER = 1.3  # Widen for recent jumps > 3%
FV_NEAR_CLOSE_MULTIPLIER = 1.4  # Widen when time_to_close < 72h

# Quote sizing
BASE_QUOTE_SIZE_USD = 50.0  # $50 base quote size
INVENTORY_SIZE_REDUCTION_MAX = 0.7  # Reduce up to 70% at full inventory

# Inventory skew (CRITICAL FIX #4: corrected sign)
INVENTORY_SKEW_MAX_CENTS = 0.01  # ±1¢ for full inventory

# Price bounds
MIN_PRICE = 0.01
MAX_PRICE = 0.99

# Tick size (standard)
STANDARD_TICK_SIZE = 0.01

# ============================================================================
# MARKOUT/TOXICITY MODULE (CRITICAL FIX #4)
# ============================================================================

# Markout intervals (seconds)
MARKOUT_INTERVALS = [30, 120, 600]  # 30s, 2m, 10m

# Toxic detection
TOXIC_MARKOUT_MEAN_VETO = -0.002  # Veto if mean < -0.2¢
TOXIC_MARKOUT_MILD = -0.001  # Widen band / reduce size if < -0.1¢
TOXIC_MARKOUT_WINDOW_SIZE = 100  # Rolling window size

# Adjustments for mild toxicity
TOXIC_FV_BAND_MULTIPLIER = 1.5  # Widen band by 50%
TOXIC_SIZE_MULTIPLIER = 0.5  # Reduce size by 50%

# ============================================================================
# CONSISTENCY/COUPLING ARBITRAGE (CRITICAL FIX #2, #7)
# ============================================================================

# Parity arb
PARITY_MIN_PROFIT = 0.005  # 0.5¢+ edge after fees

# Duplicate markets
DUPLICATE_SIMILARITY_THRESHOLD = 0.95  # 95%+ semantic similarity
DUPLICATE_PRICE_DIVERGENCE_MIN = 0.05  # 5¢+ divergence

# ============================================================================
# SATELLITE FILTER (Astra V2 Integration)
# ============================================================================

# High-conviction gates
SATELLITE_MIN_EDGE = 0.15  # 15% mispricing
SATELLITE_MIN_ROBUSTNESS = 4  # 4/5 robustness score
SATELLITE_MIN_LIQUIDITY = 1000.0  # $1000 min liquidity

# Required evidence
SATELLITE_REQUIRE_TIER_A_OR_B = True

# ============================================================================
# MONITORING & REPORTING
# ============================================================================

# Scanning intervals
SCAN_INTERVAL_SECONDS = 300  # 5 minutes for normal mode
FAST_SCAN_INTERVAL_SECONDS = 60  # 1 minute for --fast mode

# Scanner configuration (Astra V2 legacy)
MAX_CLAUDE_CALLS_PER_SCAN = 50  # Limit Claude API calls per scan
MAX_KELLY_FRACTION = 0.25  # Conservative Kelly sizing
MAX_POSITION_PCT = 0.06  # Max 6% bankroll per position
MAX_DAILY_LOSS_PCT = 0.05  # Circuit breaker at 5% daily loss
MIN_HOURS_TO_EXPIRY = 2  # Skip markets expiring soon
MIN_HOURS_TO_RESOLUTION = 48  # Skip markets resolving within 48h
MIN_MARKET_LIQUIDITY = 500  # Skip markets with < $500 liquidity
MISPRICING_THRESHOLD = 0.08  # Minimum edge to consider trading
MIN_CONFIDENCE = 0.6  # Minimum confidence for estimates
ACQUIESCENCE_CORRECTION = 0.04  # Bias correction for LLM estimates
EXTREMIZE_K = 1.25  # Extremization factor

# External data sources
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
NOAA_API_URL = "https://api.weather.gov"

# Truth Report
TRUTH_REPORT_DIR = Path(__file__).resolve().parent / "reports"
TRUTH_REPORT_DAILY = True

# ============================================================================
# MARKET FETCHING & WS SUBSCRIPTION (PAPER MODE OPTIMIZATION)
# ============================================================================

# Market fetch limits
MARKET_FETCH_LIMIT_PROD = 200  # Production: conservative
MARKET_FETCH_LIMIT_PAPER = 500  # Paper mode: fetch more for better NORMAL state coverage

# WS subscription prefiltering
MAX_SUBSCRIBE_MARKETS = 250  # Subscribe to top N markets (500 assets YES+NO)
PREFILTER_EXCLUDE_CLOSE_WINDOW = True  # Exclude markets with time_to_close < 24h before subscribe

# WS warmup (Cycle 1 only)
WS_WARMUP_TIMEOUT_S = 10  # Max warmup wait time
WS_WARMUP_MIN_BOOKS = 25  # Minimum books required before first QS computation

# Logging
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_LEVEL = "INFO"

# ============================================================================
# GATE THRESHOLDS (CRITICAL FIX #8)
# ============================================================================

# Gate A: Research → Paper Trading
GATE_A_SHARPE_MIN = 1.2
GATE_A_COST_STRESS_MULTIPLIER = 1.5

# Gate B: Paper → Small Live ($500)
GATE_B_MIN_DURATION_DAYS = 7  # 7-14 days
GATE_B_MIN_FILLS = 3000  # 3,000+ fills
GATE_B_MIN_CLUSTERS = 8  # ≥8 clusters traded
GATE_B_MAX_CLUSTER_CONCENTRATION = 0.20  # No cluster >20% of fills
GATE_B_REALIZED_SPREAD_MIN = 0.0  # Median > 0
GATE_B_MARKOUT_TTEST_PVALUE = 0.05  # Not significantly negative
GATE_B_FILL_RATE_TOLERANCE = 0.20  # Within 20% of assumptions
GATE_B_TOP_MARKETS_MAX_PCT = 0.30  # Top 5 markets < 30% of P&L
GATE_B_SHARPE_MIN = 1.2

# Gate C: Scale ($500 → $5000)
GATE_C_SHARPE_MIN_90D = 2.0
GATE_C_CALMAR_MIN = 2.0
GATE_C_CALMAR_DURATION_DAYS = 30
GATE_C_MAX_DRAWDOWN = -0.15  # -15%

# ============================================================================
# CANONICAL MID CALCULATION (CRITICAL FIX: Final ChatGPT)
# ============================================================================

# Fallback rules for one-sided books
MID_FALLBACK_ONE_SIDED_OFFSET = 0.02  # ±2¢ from single side
MID_FALLBACK_STALE_AGE_MS = 10000  # 10s max age

# ============================================================================
# CLUSTER ASSIGNMENT (CRITICAL FIX: Final ChatGPT)
# ============================================================================

# Determinism requirement: same market text → same cluster_id
CLUSTER_CACHE_FILE = Path(__file__).resolve().parent / "memory" / "cluster_cache.json"

# ============================================================================
# NEGATIVE RISK EVENTS (CRITICAL FIX #17)
# ============================================================================

# negRisk mechanics
DISABLE_PARITY_FOR_NEG_RISK = True  # Disable parity scans for negRisk events
NEG_RISK_SINGLE_CLUSTER = True  # Treat all markets in negRisk event as one cluster

# ============================================================================
# WALLET PREFLIGHT CHECKS (CRITICAL FIX #19)
# ============================================================================

# Minimum balances
MIN_USDC_BALANCE = 100.0  # $100 minimum to start

# ============================================================================
# PATHS
# ============================================================================

# Memory
MEMORY_DIR = Path(__file__).resolve().parent / "memory"
PAPER_POSITIONS_FILE = MEMORY_DIR / "paper_positions.json"
PREDICTIONS_FILE = MEMORY_DIR / "predictions.json"
PRICE_SNAPSHOTS_FILE = MEMORY_DIR / "price_snapshots.json"
TRADES_DB_FILE = MEMORY_DIR / "astra_trades.db"

# Data
DATA_DIR = Path(__file__).resolve().parent / "data"

# Create directories
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRUTH_REPORT_DIR.mkdir(parents=True, exist_ok=True)
