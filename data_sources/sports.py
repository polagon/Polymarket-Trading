"""
Sports probability estimation via The Odds API.

Fetches live bookmaker odds for upcoming games and deviggs them to
produce a fair-value probability estimate for Polymarket sports markets.

API: https://the-odds-api.com  (free tier: 500 requests/month)
Key: configured via ODDS_API_KEY in .env / config.py

Devigging (removing the vig/overround):
  Bookmakers set odds that sum > 1 to guarantee profit.
  To get a fair probability we normalize:
    p_fair = p_implied / sum(all_implied_probs)

Usage:
  from data_sources.sports import match_sports_market, SportsEstimate
  estimate = await match_sports_market(market_question)
"""
import json
import logging
import re
import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import ODDS_API_KEY

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
logger = logging.getLogger("astra.sports")

# Sports IDs supported by The Odds API
# fmt: off
SPORT_KEYS = [
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "basketball_nba",
    "basketball_ncaab",
    "baseball_mlb",
    "icehockey_nhl",
    "soccer_epl",               # English Premier League
    "soccer_uefa_champs_league",
    "soccer_usa_mls",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "tennis_atp_us_open",
    "tennis_wta_us_open",
    "mma_mixed_martial_arts",   # UFC
]
# fmt: on

# Cache odds data to avoid hammering the API (1 request = 1 API credit)
_odds_cache: dict[str, tuple[float, list]] = {}  # sport_key → (timestamp, events)
CACHE_TTL_SECONDS = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Quota tracker — prevents silent burnout of 500 req/month free tier
# ---------------------------------------------------------------------------
_QUOTA_FILE = Path("memory/odds_api_usage.json")
_MONTHLY_QUOTA = 500
_QUOTA_WARN_THRESHOLD = 450   # Warn when < 50 remaining


def _load_quota() -> dict:
    """Load quota usage from disk. Returns dict with 'month' and 'used' keys."""
    if _QUOTA_FILE.exists():
        try:
            return json.loads(_QUOTA_FILE.read_text())
        except Exception:
            pass
    return {"month": "", "used": 0}


def _save_quota(data: dict):
    Path("memory").mkdir(exist_ok=True)
    try:
        _QUOTA_FILE.write_text(json.dumps(data))
    except Exception as e:
        logger.warning("Could not save quota tracker: %s", e)


def _track_requests(count: int) -> int:
    """
    Record 'count' API requests used. Returns remaining quota for this month.
    Auto-resets on the 1st of each calendar month.
    """
    now = datetime.now(timezone.utc)
    current_month = now.strftime("%Y-%m")
    data = _load_quota()

    if data["month"] != current_month:
        # New month — reset counter
        logger.info("Odds API quota reset for new month: %s", current_month)
        data = {"month": current_month, "used": 0}

    data["used"] += count
    _save_quota(data)

    remaining = _MONTHLY_QUOTA - data["used"]
    if remaining < (_MONTHLY_QUOTA - _QUOTA_WARN_THRESHOLD):
        logger.warning(
            "Odds API quota: %d/%d used this month (%d remaining)",
            data["used"], _MONTHLY_QUOTA, remaining
        )
    return remaining


def get_quota_status() -> dict:
    """Return current quota usage for external monitoring."""
    now = datetime.now(timezone.utc)
    current_month = now.strftime("%Y-%m")
    data = _load_quota()
    if data["month"] != current_month:
        return {"month": current_month, "used": 0, "remaining": _MONTHLY_QUOTA, "pct_used": 0.0}
    used = data["used"]
    remaining = max(0, _MONTHLY_QUOTA - used)
    return {
        "month": current_month,
        "used": used,
        "remaining": remaining,
        "pct_used": round(used / _MONTHLY_QUOTA * 100, 1),
    }


@dataclass
class TeamOdds:
    name: str
    bookmaker_count: int
    avg_implied_prob: float   # Raw bookmaker implied probability (with vig)
    devigged_prob: float      # Fair probability after removing vig
    consensus_american: int   # American odds equivalent


@dataclass
class GameOdds:
    """Devigged fair-value odds for a single game."""
    sport: str
    home_team: str
    away_team: str
    commence_time: str        # ISO8601 UTC
    hours_to_game: float
    home: TeamOdds
    away: TeamOdds
    draw: Optional[TeamOdds]  # Only for soccer/some sports
    bookmaker_count: int
    market_type: str          # "h2h" (moneyline)


@dataclass
class SportsEstimate:
    """Probability estimate for a Polymarket sports question."""
    market_question: str
    matched_game: Optional[str]   # "TeamA vs TeamB"
    probability: float            # Fair p(YES) for the Polymarket question
    confidence: float             # 0-1 data quality score
    source: str                   # "the_odds_api"
    reasoning: str
    hours_to_game: float


async def _fetch_sport_odds(sport_key: str, session: aiohttp.ClientSession) -> list:
    """Fetch h2h odds for a sport, using cache."""
    now = datetime.now(timezone.utc).timestamp()
    cached = _odds_cache.get(sport_key)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        async with session.get(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us,uk,eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                _odds_cache[sport_key] = (now, data)
                # Sync quota with authoritative API header value
                remaining_header = resp.headers.get("x-requests-remaining")
                if remaining_header:
                    try:
                        api_remaining = int(remaining_header)
                        # Backfill quota file with authoritative value from API
                        quota_data = _load_quota()
                        now_month = datetime.now(timezone.utc).strftime("%Y-%m")
                        if quota_data.get("month") == now_month:
                            authoritative_used = _MONTHLY_QUOTA - api_remaining
                            if authoritative_used > quota_data.get("used", 0):
                                quota_data["used"] = authoritative_used
                                _save_quota(quota_data)
                        if api_remaining < 50:
                            logger.warning("Odds API quota critically low: %d requests remaining", api_remaining)
                    except ValueError:
                        pass
                return data
            elif resp.status == 401:
                logger.error("Odds API: Bad API key (401) for sport %s", sport_key)
                return []
            elif resp.status == 422:
                return []  # Sport not currently active / off-season
            elif resp.status == 429:
                logger.warning("Odds API: Rate limited (429) for sport %s", sport_key)
                return []
    except Exception as e:
        logger.debug("Odds API fetch failed for %s: %s", sport_key, e)
    return []


def _devig_game(event: dict) -> Optional[GameOdds]:
    """Extract and devig odds from a single Odds API event."""
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    sport = event.get("sport_key", "")
    commence = event.get("commence_time", "")

    if not home or not away:
        return None

    # Parse hours to game
    try:
        game_time = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        hours_to_game = (game_time - datetime.now(timezone.utc)).total_seconds() / 3600
    except Exception:
        hours_to_game = 24.0

    # Collect all bookmaker odds
    home_odds_list = []
    away_odds_list = []
    draw_odds_list = []

    bookmakers = event.get("bookmakers", [])
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = float(outcome.get("price", 1.0))
                if price <= 1.0:
                    continue
                implied = 1.0 / price
                if name.lower() == home.lower():
                    home_odds_list.append(implied)
                elif name.lower() == away.lower():
                    away_odds_list.append(implied)
                elif name.lower() in ("draw", "tie"):
                    draw_odds_list.append(implied)

    if not home_odds_list or not away_odds_list:
        return None

    # Average across bookmakers
    home_avg = sum(home_odds_list) / len(home_odds_list)
    away_avg = sum(away_odds_list) / len(away_odds_list)
    draw_avg = sum(draw_odds_list) / len(draw_odds_list) if draw_odds_list else 0.0

    total_implied = home_avg + away_avg + draw_avg

    # Devig
    home_fair = home_avg / total_implied
    away_fair = away_avg / total_implied
    draw_fair = draw_avg / total_implied if draw_avg > 0 else 0.0

    def to_american(p: float) -> int:
        if p <= 0 or p >= 1:
            return 0
        if p >= 0.5:
            return -round((p / (1 - p)) * 100)
        else:
            return round(((1 - p) / p) * 100)

    n_bm = len(bookmakers)

    home_team_odds = TeamOdds(
        name=home,
        bookmaker_count=len(home_odds_list),
        avg_implied_prob=home_avg,
        devigged_prob=home_fair,
        consensus_american=to_american(home_fair),
    )
    away_team_odds = TeamOdds(
        name=away,
        bookmaker_count=len(away_odds_list),
        avg_implied_prob=away_avg,
        devigged_prob=away_fair,
        consensus_american=to_american(away_fair),
    )
    draw_team_odds = None
    if draw_fair > 0:
        draw_team_odds = TeamOdds(
            name="Draw",
            bookmaker_count=len(draw_odds_list),
            avg_implied_prob=draw_avg,
            devigged_prob=draw_fair,
            consensus_american=to_american(draw_fair),
        )

    return GameOdds(
        sport=sport,
        home_team=home,
        away_team=away,
        commence_time=commence,
        hours_to_game=hours_to_game,
        home=home_team_odds,
        away=away_team_odds,
        draw=draw_team_odds,
        bookmaker_count=n_bm,
        market_type="h2h",
    )


def _normalize_team_name(name: str) -> str:
    """Lowercase, strip punctuation, remove common suffixes for fuzzy matching."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    # Remove common city/location prefixes to match "Lakers" vs "Los Angeles Lakers"
    stop_words = {"fc", "city", "united", "sporting", "club", "athletics"}
    parts = [p for p in name.split() if p not in stop_words or len(name.split()) <= 2]
    return " ".join(parts)


def _team_match_score(query: str, candidate: str) -> float:
    """Return 0-1 similarity between query and candidate team name."""
    q = _normalize_team_name(query)
    c = _normalize_team_name(candidate)
    if q == c:
        return 1.0
    if q in c or c in q:
        return 0.85
    # Word overlap
    q_words = set(q.split())
    c_words = set(c.split())
    overlap = q_words & c_words
    if overlap:
        return len(overlap) / max(len(q_words), len(c_words))
    return 0.0


def _parse_teams_from_question(question: str) -> tuple[str, str, str]:
    """
    Extract team names and implied bet direction from a Polymarket question.

    Returns: (team1, team2, outcome_type)
    outcome_type: "home_win", "away_win", "team_win", "cover", "over_under"
    """
    q = question.strip()

    # "Will [Team A] beat [Team B]?" or "Will [Team A] defeat [Team B]?"
    m = re.search(
        r"will\s+(.+?)\s+(?:beat|defeat|win (?:against|over|vs))\s+(.+?)(?:\?|$)",
        q, re.IGNORECASE
    )
    if m:
        return m.group(1).strip(), m.group(2).strip(".?").strip(), "team_win"

    # "[Team A] vs [Team B]" style
    m = re.search(r"(.+?)\s+(?:vs?\.?|versus|@)\s+(.+?)(?:\?|\s+–|\s+-|$)", q, re.IGNORECASE)
    if m:
        t1 = m.group(1).strip()
        t2 = m.group(2).strip().rstrip("?").strip()
        # If question asks "who wins" or "winner", outcome is ambiguous — return both
        return t1, t2, "either_wins"

    # "Will [Team] win [event/game/match]?"
    m = re.search(r"will\s+(.+?)\s+win\b", q, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip("?"), "", "team_win"

    return "", "", ""


def _find_best_game(
    team1: str,
    team2: str,
    outcome_type: str,
    all_games: list[GameOdds],
) -> tuple[Optional[GameOdds], float, str]:
    """
    Find the best matching game and compute p(YES) for the Polymarket question.

    Returns: (game, probability, reasoning)
    """
    if not team1:
        return None, 0.5, "Could not parse team names from question."

    best_game = None
    best_score = 0.0
    best_team1_is_home = True

    for game in all_games:
        # Skip games too far out (>7 days) or already started (<-2h)
        if game.hours_to_game > 168 or game.hours_to_game < -2:
            continue

        h_score = _team_match_score(team1, game.home_team)
        a_score = _team_match_score(team1, game.away_team)

        if team2:
            h2_score = _team_match_score(team2, game.away_team)
            a2_score = _team_match_score(team2, game.home_team)
            # Both teams must match
            score_home_team1 = min(h_score, h2_score)
            score_away_team1 = min(a_score, a2_score)
        else:
            score_home_team1 = h_score
            score_away_team1 = a_score

        score = max(score_home_team1, score_away_team1)
        team1_is_home = score_home_team1 >= score_away_team1

        if score > best_score:
            best_score = score
            best_game = game
            best_team1_is_home = team1_is_home

    if best_game is None or best_score < 0.4:
        return None, 0.5, f"No matching game found for '{team1}' (best score: {best_score:.2f})"

    game = best_game
    team1_odds = game.home if best_team1_is_home else game.away
    team2_odds = game.away if best_team1_is_home else game.home

    if outcome_type == "team_win":
        # Question asks if team1 wins
        prob = team1_odds.devigged_prob
        reasoning = (
            f"Matched '{team1}' → {team1_odds.name} ({game.home_team} vs {game.away_team}, "
            f"{game.hours_to_game:.1f}h). "
            f"Consensus devigged win prob: {prob:.1%} "
            f"(from {game.bookmaker_count} bookmakers, {team1_odds.bookmaker_count} pricing this team). "
            f"Opponent devigged: {team2_odds.devigged_prob:.1%}."
        )
    elif outcome_type == "either_wins":
        # Question format "TeamA vs TeamB" — try to figure out from context
        # Default: report home team win probability (team1's position in the question)
        prob = team1_odds.devigged_prob
        reasoning = (
            f"Matched '{team1}' vs '{team2}' → {game.home_team} vs {game.away_team} "
            f"({game.hours_to_game:.1f}h). "
            f"p({team1_odds.name} wins) = {prob:.1%} "
            f"(devigged from {game.bookmaker_count} books). "
            f"p({team2_odds.name} wins) = {team2_odds.devigged_prob:.1%}."
        )
    else:
        prob = team1_odds.devigged_prob
        reasoning = (
            f"Matched '{team1}' → {team1_odds.name} "
            f"devigged win prob: {prob:.1%}."
        )

    return game, prob, reasoning


async def fetch_all_odds(sport_keys: Optional[list[str]] = None) -> list[GameOdds]:
    """
    Fetch and devig odds for all supported sports. Cached per CACHE_TTL_SECONDS.

    Args:
        sport_keys: Subset of sports to fetch (default: all SPORT_KEYS).
                    Use to reduce quota burn when only certain sports are relevant.
    """
    if not ODDS_API_KEY:
        return []

    # Quota guard: skip if < 50 requests remaining this month
    quota = get_quota_status()
    if quota["remaining"] < 50:
        logger.warning(
            "Odds API quota nearly exhausted (%d remaining) — skipping sports fetch",
            quota["remaining"],
        )
        return []

    keys_to_fetch = sport_keys if sport_keys is not None else SPORT_KEYS

    # Count only cache-misses against quota (cached fetches don't use credits)
    now = datetime.now(timezone.utc).timestamp()
    fresh_fetches = sum(
        1 for sk in keys_to_fetch
        if not (_odds_cache.get(sk) and now - _odds_cache[sk][0] < CACHE_TTL_SECONDS)
    )
    if fresh_fetches > 0:
        _track_requests(fresh_fetches)

    all_games: list[GameOdds] = []
    # Use a semaphore to limit concurrent requests (avoids 429 rate limits)
    # The Odds API rate-limits burst requests; 3 concurrent + stagger avoids it
    sem = asyncio.Semaphore(3)  # max 3 concurrent requests

    async def fetch_with_sem(sk: str, session: aiohttp.ClientSession) -> list:
        async with sem:
            result = await _fetch_sport_odds(sk, session)
            if result:   # small delay after each successful fetch to avoid burst 429
                await asyncio.sleep(0.1)
            return result

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_sem(sk, session) for sk in keys_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for events in results:
            if isinstance(events, Exception) or not events:
                continue
            for event in events:
                game = _devig_game(event)
                if game:
                    all_games.append(game)

    return all_games


async def match_sports_market(question: str) -> SportsEstimate:
    """
    Match a Polymarket sports question to a real game and estimate p(YES).

    Returns a SportsEstimate with probability and reasoning.
    If no match is found, returns confidence=0 (signal will be ignored).
    """
    team1, team2, outcome_type = _parse_teams_from_question(question)

    if not team1 and not team2:
        return SportsEstimate(
            market_question=question,
            matched_game=None,
            probability=0.5,
            confidence=0.0,
            source="the_odds_api",
            reasoning="Could not identify team names in question.",
            hours_to_game=0.0,
        )

    if not ODDS_API_KEY:
        return SportsEstimate(
            market_question=question,
            matched_game=None,
            probability=0.5,
            confidence=0.0,
            source="the_odds_api",
            reasoning="ODDS_API_KEY not configured.",
            hours_to_game=0.0,
        )

    all_games = await fetch_all_odds()
    game, prob, reasoning = _find_best_game(team1, team2, outcome_type, all_games)

    if game is None:
        return SportsEstimate(
            market_question=question,
            matched_game=None,
            probability=0.5,
            confidence=0.0,
            source="the_odds_api",
            reasoning=reasoning,
            hours_to_game=0.0,
        )

    # Confidence: higher when more bookmakers agree and game is near
    confidence = min(0.85, 0.5 + (game.bookmaker_count / 30) * 0.35)
    if game.hours_to_game < 4:
        confidence = min(confidence + 0.1, 0.92)  # Very close to game — sharper lines

    return SportsEstimate(
        market_question=question,
        matched_game=f"{game.home_team} vs {game.away_team}",
        probability=prob,
        confidence=confidence,
        source="the_odds_api",
        reasoning=reasoning,
        hours_to_game=game.hours_to_game,
    )


def _infer_sport_keys_from_questions(questions: list[str]) -> list[str]:
    """
    Precision fetching: infer which sport types are relevant from question text.
    Only fetch those leagues — avoids burning quota on irrelevant sports.

    Example: ["Will Chiefs beat Ravens?"] → ["americanfootball_nfl"]
    vs fetching all 16 sport types blindly.
    """
    q_text = " ".join(questions).lower()
    relevant = []

    sport_signals = {
        "americanfootball_nfl": ["nfl", "super bowl", "chiefs", "patriots", "eagles", "cowboys",
                                  "49ers", "ravens", "bills", "bengals", "packers", "quarterback"],
        "americanfootball_ncaaf": ["ncaaf", "college football", "cfp", "ncaa football"],
        "basketball_nba": ["nba", "celtics", "lakers", "warriors", "bucks", "heat", "knicks",
                           "nuggets", "clippers", "nets", "playoff", "finals", "mvp"],
        "basketball_ncaab": ["ncaab", "march madness", "ncaa tournament", "college basketball"],
        "baseball_mlb": ["mlb", "world series", "yankees", "dodgers", "red sox", "cubs",
                         "mets", "astros", "braves", "pennant"],
        "icehockey_nhl": ["nhl", "stanley cup", "maple leafs", "rangers", "bruins",
                          "penguins", "oilers", "avalanche", "golden knights"],
        "soccer_epl": ["premier league", "epl", "chelsea", "arsenal", "manchester", "liverpool",
                       "tottenham", "everton", "leicester"],
        "soccer_uefa_champs_league": ["champions league", "ucl", "real madrid", "barcelona",
                                       "psg", "juventus", "bayern", "inter milan"],
        "soccer_usa_mls": ["mls", "lafc", "atlanta united", "seattle sounders", "galaxy"],
        "soccer_spain_la_liga": ["la liga", "laliga", "real madrid", "barcelona", "atletico"],
        "soccer_germany_bundesliga": ["bundesliga", "dortmund", "bayer leverkusen"],
        "soccer_italy_serie_a": ["serie a", "juventus", "inter milan", "ac milan", "roma"],
        "soccer_france_ligue_one": ["ligue 1", "ligue one", "psg", "paris saint-germain"],
        "tennis_atp_us_open": ["tennis", "atp", "us open", "djokovic", "alcaraz", "sinner"],
        "tennis_wta_us_open": ["wta", "swiatek", "sabalenka", "gauff", "wimbledon"],
        "mma_mixed_martial_arts": ["ufc", "mma", "conor mcgregor", "khabib", "jon jones",
                                   "poirier", "adesanya", "izzy"],
    }

    for sport_key, signals in sport_signals.items():
        if any(signal in q_text for signal in signals):
            relevant.append(sport_key)

    # Fallback: if nothing matched but there are sports questions, fetch the most common
    if not relevant and questions:
        relevant = ["americanfootball_nfl", "basketball_nba", "baseball_mlb", "icehockey_nhl"]
        logger.debug("No sport keywords matched — fetching top 4 by default (%d questions)", len(questions))
    else:
        logger.info("Precision sport fetch: %d leagues for %d questions (saved %d API calls)",
                    len(relevant), len(questions), len(SPORT_KEYS) - len(relevant))

    return relevant


async def get_sports_estimates(
    questions: list[str],
) -> dict[str, SportsEstimate]:
    """
    Batch-estimate probabilities for a list of sports market questions.
    Uses precision fetching to only query sport leagues relevant to the questions.
    """
    if not ODDS_API_KEY or not questions:
        return {}

    # Precision fetch: only the sport types that match active questions
    relevant_sport_keys = _infer_sport_keys_from_questions(questions)
    all_games = await fetch_all_odds(sport_keys=relevant_sport_keys)
    results = {}

    for q in questions:
        team1, team2, outcome_type = _parse_teams_from_question(q)
        game, prob, reasoning = _find_best_game(team1, team2, outcome_type, all_games)

        if game is None:
            results[q] = SportsEstimate(
                market_question=q,
                matched_game=None,
                probability=0.5,
                confidence=0.0,
                source="the_odds_api",
                reasoning=reasoning,
                hours_to_game=0.0,
            )
        else:
            confidence = min(0.85, 0.5 + (game.bookmaker_count / 30) * 0.35)
            if game.hours_to_game < 4:
                confidence = min(confidence + 0.1, 0.92)
            results[q] = SportsEstimate(
                market_question=q,
                matched_game=f"{game.home_team} vs {game.away_team}",
                probability=prob,
                confidence=confidence,
                source="the_odds_api",
                reasoning=reasoning,
                hours_to_game=game.hours_to_game,
            )

    return results
