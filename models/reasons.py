"""
Centralized reason enums for all gates and vetoes — single source of truth.

Every module that emits a reason string MUST import from here.
Tests MUST assert exact enum values (not partial matches).

Loop 4: Allocator-Grade Tradability Contract.
"""

# ── Definition Gate ──
REASON_NO_DEFINITION = "no_definition_contract"
REASON_DEFINITION_INCOMPLETE = "definition_incomplete"  # append: "|".join(missing)
REASON_DEFINITION_OK = "definition_ok"

# ── EV Gate ──
REASON_EV_INVALID_PROB_BOUNDS = "ev_veto: invalid_prob_bounds"
REASON_EV_INVALID_MARKET_PRICE = "ev_veto: invalid_market_price"
REASON_EV_NET_LB_BELOW_THRESHOLD = "ev_veto: net_lb<=threshold"
REASON_EV_MAKER_ONLY = "ev_ok: maker_only"

# ── Risk Engine ──
REASON_RISK_COOLDOWN = "risk_veto: cooldown_active"
REASON_RISK_DAILY_LOSS = "risk_veto: halted_daily_loss"
REASON_RISK_DRAWDOWN = "risk_veto: halted_drawdown"
REASON_RISK_BROKEN_REGIME = "risk_veto: broken_regime"
REASON_RISK_CATEGORY_CAP = "risk_veto: category_cap"
REASON_RISK_PORTFOLIO_CAP = "risk_veto: portfolio_cap"
REASON_RISK_OK = "risk_ok"

# ── Lint (stable missing semantics) ──
LINT_MISSING_CONDITION_OP = "missing_condition_op"
LINT_MISSING_CONDITION_LEVEL = "missing_condition_level"
LINT_INVALID_CONDITION_LEVEL_TYPE = "invalid_condition_level_type"
LINT_INVALID_CONDITION_WINDOW = "invalid_condition_window"
LINT_UNKNOWN_CONDITION_KEY = "unknown_condition_key"  # append: ": {key}"
LINT_MISSING_MEASUREMENT_TIME = "missing_measurement_time"
LINT_MISSING_ORACLE_FEED = "missing_oracle_feed"
LINT_MISSING_ORACLE_ROUNDING = "missing_oracle_rounding"
LINT_INVALID_ORACLE_ROUNDING = "invalid_oracle_rounding"
LINT_MISSING_ORACLE_FINALITY = "missing_oracle_finality"
LINT_UNKNOWN_ORACLE_KEY = "unknown_oracle_key"  # append: ": {key}"
LINT_INVALID_CUTOFF = "invalid_cutoff_ts_utc"

# Allowed rounding rules for crypto_threshold
ALLOWED_ROUNDING_RULES = frozenset({"floor_int", "round_int", "ceil_int", "floor_cent", "round_cent"})

# Allowed condition operators
ALLOWED_CONDITION_OPS = frozenset({">=", ">", "<=", "<", "=="})

# ── Order Manager ──
REASON_ORDER_CROSS_REJECTED = "order_rejected: would_cross"
REASON_ORDER_CHASE_EXCEEDED = "order_rejected: chase_exceeded"
REASON_ORDER_TOXIC_NO_IMPROVE = "order_rejected: toxic_no_improve"
REASON_ORDER_TTL_EXPIRED = "order_cancel: ttl_expired"
REASON_ORDER_STALE = "order_cancel: stale"
