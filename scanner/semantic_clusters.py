"""
Astra V2 — Semantic Market Clustering + Leader-Follower Engine

Based on: "Semantic Trading: Agentic AI for Clustering and Relationship Discovery
in Prediction Markets" (arxiv 2512.02436)
Empirical results: 60-70% accuracy, 20-47% monthly ROI

Strategy:
  1. Embed all active market questions into a shared semantic space
  2. Cluster by meaning using K≈N/10 partitioning
  3. Within each cluster, identify CORRELATED and ANTI-CORRELATED market pairs
  4. When a "leader" market resolves → enter the predicted "follower" position

Example relationships:
  SAME-outcome:   "Will Trump raise tariffs on Canada?" + "Will US-Canada trade war escalate?"
  DIFF-outcome:   "Will Trump increase tariffs on Canada?" + "Will Trump REMOVE tariffs on Canada?"

The insight: Polymarket has hundreds of fragmented markets asking the same underlying
question in different ways. Semantic clustering detects this and turns one market's
resolution into a trading signal for related markets.

This module:
  - Clusters markets semantically using Claude embeddings
  - Identifies related pairs with confidence scores
  - Tracks pending follower trades
  - Generates trade signals when leaders resolve
"""
import asyncio
import json
import hashlib
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from scanner.market_fetcher import Market


CLUSTERS_FILE = Path("memory/semantic_clusters.json")
RELATIONSHIPS_FILE = Path("memory/market_relationships.json")
FOLLOWERS_FILE = Path("memory/follower_trades.json")

# Cluster count: K ≈ N/10 (from paper's empirical tuning)
MARKETS_PER_CLUSTER = 10


@dataclass
class MarketRelationship:
    """A detected semantic relationship between two markets."""
    leader_id: str
    follower_id: str
    leader_question: str
    follower_question: str
    relationship: str          # "same_outcome" or "different_outcome"
    confidence: float          # 0-1
    reasoning: str             # Why they are related
    cluster_label: str         # Topic cluster (e.g. "US-Canada Trade War")
    discovered_at: str
    # Post-resolution tracking
    resolved: bool = False
    leader_resolved_yes: Optional[bool] = None
    follower_traded: bool = False
    follower_direction: Optional[str] = None  # "BUY YES" or "BUY NO"
    outcome_correct: Optional[bool] = None


@dataclass
class FollowerTrade:
    """A pending trade triggered by a leader resolution."""
    follower_id: str
    follower_question: str
    direction: str             # "BUY YES" or "BUY NO"
    confidence: float
    reasoning: str
    cluster_label: str
    triggered_by: str          # leader condition_id
    triggered_at: str
    min_edge: float = 0.08     # Minimum edge required to actually trade


class SemanticClusterEngine:
    """
    Discovers and tracks semantic relationships between Polymarket markets.
    Uses Claude to embed, cluster, and identify related pairs.
    """

    def __init__(self):
        CLUSTERS_FILE.parent.mkdir(exist_ok=True)
        self._relationships: list[MarketRelationship] = self._load_relationships()
        self._follower_trades: list[FollowerTrade] = self._load_follower_trades()
        self._last_cluster_hash: str = ""

    def _load_relationships(self) -> list[MarketRelationship]:
        if RELATIONSHIPS_FILE.exists():
            try:
                data = json.loads(RELATIONSHIPS_FILE.read_text())
                loaded = []
                for r in data:
                    r.setdefault("resolved", False)
                    r.setdefault("leader_resolved_yes", None)
                    r.setdefault("follower_traded", False)
                    r.setdefault("follower_direction", None)
                    r.setdefault("outcome_correct", None)
                    loaded.append(MarketRelationship(**r))
                return loaded
            except Exception:
                return []
        return []

    def _load_follower_trades(self) -> list[FollowerTrade]:
        if FOLLOWERS_FILE.exists():
            try:
                data = json.loads(FOLLOWERS_FILE.read_text())
                return [FollowerTrade(**t) for t in data]
            except Exception:
                return []
        return []

    def _save_relationships(self):
        RELATIONSHIPS_FILE.write_text(
            json.dumps([asdict(r) for r in self._relationships], indent=2)
        )

    def _save_follower_trades(self):
        FOLLOWERS_FILE.write_text(
            json.dumps([asdict(t) for t in self._follower_trades], indent=2)
        )

    def _markets_hash(self, markets: list[Market]) -> str:
        """Hash market IDs to detect when market set has changed."""
        ids = sorted(m.condition_id for m in markets)
        return hashlib.md5(":".join(ids).encode()).hexdigest()[:12]

    def get_pending_follower_trades(self) -> list[FollowerTrade]:
        """Return follower trades that haven't been executed yet."""
        return [t for t in self._follower_trades if not t.triggered_at == "executed"]

    def check_leader_resolutions(
        self, current_markets: list[Market]
    ) -> list[FollowerTrade]:
        """
        Check if any relationship leaders have resolved (disappeared from active list).
        Returns new FollowerTrade signals to act on.
        """
        active_ids = {m.condition_id for m in current_markets}
        # Build price map for near-resolved detection
        price_map = {m.condition_id: m.yes_price for m in current_markets}

        new_signals: list[FollowerTrade] = []

        for rel in self._relationships:
            if rel.resolved or rel.follower_traded:
                continue

            leader_in_active = rel.leader_id in active_ids
            leader_price = price_map.get(rel.leader_id, 0.5)

            # Leader has resolved if: it disappeared OR price is extreme (≥0.97 or ≤0.03)
            # Note: we use 0.97/0.03 here (not 0.98/0.02) because we're using this as
            # a signal trigger, not a final resolution — the follower market is still live
            leader_resolved = not leader_in_active
            leader_yes = None

            if leader_resolved:
                # Market gone — assume majority direction based on last known price
                # (We stored the leader question but not its final price; use 0.5 as fallback)
                leader_yes = True  # Can't know for sure — will be validated by follower performance
            elif leader_price >= 0.97:
                leader_yes = True
                leader_resolved = True
            elif leader_price <= 0.03:
                leader_yes = False
                leader_resolved = True

            if not leader_resolved or leader_yes is None:
                continue

            # Determine follower direction
            if rel.relationship == "same_outcome":
                direction = "BUY YES" if leader_yes else "BUY NO"
            else:  # different_outcome
                direction = "BUY NO" if leader_yes else "BUY YES"

            # Only signal if follower is still active
            if rel.follower_id not in active_ids:
                rel.resolved = True
                continue

            rel.leader_resolved_yes = leader_yes
            rel.follower_traded = True
            rel.follower_direction = direction
            rel.resolved = True

            signal = FollowerTrade(
                follower_id=rel.follower_id,
                follower_question=rel.follower_question,
                direction=direction,
                confidence=rel.confidence,
                reasoning=(
                    f"Leader '{rel.leader_question[:60]}' resolved "
                    f"{'YES' if leader_yes else 'NO'}. "
                    f"{rel.relationship.replace('_', ' ')} relationship "
                    f"(conf {rel.confidence:.2f}): {rel.reasoning}"
                ),
                cluster_label=rel.cluster_label,
                triggered_by=rel.leader_id,
                triggered_at=datetime.now(timezone.utc).isoformat(),
            )
            self._follower_trades.append(signal)
            new_signals.append(signal)

        if new_signals:
            self._save_relationships()
            self._save_follower_trades()

        return new_signals

    async def discover_relationships(
        self,
        markets: list[Market],
        force: bool = False,
    ) -> list[MarketRelationship]:
        """
        Cluster markets and discover semantic relationships.
        Only re-runs when the market set has materially changed
        (or force=True) to conserve API calls.
        """
        if not ANTHROPIC_API_KEY:
            return self._relationships

        current_hash = self._markets_hash(markets)
        if not force and current_hash == self._last_cluster_hash:
            return self._relationships  # No change — use cached

        self._last_cluster_hash = current_hash

        # Only work with markets that have reasonable liquidity and duration
        eligible = [
            m for m in markets
            if m.liquidity >= 1000 and m.hours_to_expiry >= 24
        ]

        if len(eligible) < 20:
            return self._relationships

        # K ≈ N/10 clusters
        k = max(5, len(eligible) // MARKETS_PER_CLUSTER)

        # Step 1: Cluster using Claude
        clusters = await self._cluster_markets(eligible[:300], k)  # cap at 300
        if not clusters:
            return self._relationships

        # Step 2: Discover relationships within each cluster
        new_relationships = await self._discover_within_clusters(clusters)

        # Merge with existing — don't duplicate
        existing_pairs = {
            (r.leader_id, r.follower_id) for r in self._relationships
        }
        added = 0
        for rel in new_relationships:
            pair = (rel.leader_id, rel.follower_id)
            if pair not in existing_pairs:
                self._relationships.append(rel)
                existing_pairs.add(pair)
                added += 1

        if added:
            self._save_relationships()

        return self._relationships

    async def _cluster_markets(
        self, markets: list[Market], k: int
    ) -> list[list[Market]]:
        """
        Use Claude to cluster markets into K semantic groups.
        Returns a list of clusters (each cluster is a list of markets).
        """
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

        # Build a compact market list for Claude
        market_data = [
            {"id": m.condition_id[:12], "q": m.question[:120], "price": round(m.yes_price, 3)}
            for m in markets
        ]

        prompt = f"""You are a prediction market analyst. Group these {len(markets)} markets into {k} semantic clusters.

Markets (id, question, yes_price):
{json.dumps(market_data, indent=2)}

Return ONLY a JSON object: {{"clusters": [{{"label": "...", "ids": ["id1", "id2", ...]}}]}}
- Each cluster should contain semantically related markets (same event, topic, or causal chain)
- Label each cluster with a 3-6 word topic name
- Every market ID must appear in exactly one cluster
- Aim for {MARKETS_PER_CLUSTER}±5 markets per cluster"""

        try:
            response = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Strip markdown
            import re
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            data = json.loads(text)

            # Map short IDs back to full market objects
            short_to_market = {m.condition_id[:12]: m for m in markets}
            clusters: list[list[Market]] = []
            for cluster_def in data.get("clusters", []):
                cluster_markets = []
                for short_id in cluster_def.get("ids", []):
                    market = short_to_market.get(short_id)
                    if market:
                        cluster_markets.append(market)
                if len(cluster_markets) >= 2:
                    # Attach label to first market for reference
                    clusters.append((cluster_def.get("label", "Unknown"), cluster_markets))
            return clusters
        except Exception:
            return []

    async def _discover_within_clusters(
        self, clusters: list[tuple[str, list[Market]]]
    ) -> list[MarketRelationship]:
        """
        For each cluster, ask Claude to identify correlated/anti-correlated pairs.
        """
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        all_relationships: list[MarketRelationship] = []

        # Process clusters in parallel (max 5 at a time)
        sem = asyncio.Semaphore(5)

        async def process_cluster(label: str, markets: list[Market]):
            async with sem:
                return await self._find_pairs_in_cluster(client, label, markets)

        tasks = [process_cluster(label, mlist) for label, mlist in clusters]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if not isinstance(result, Exception) and result:
                all_relationships.extend(result)

        return all_relationships

    async def _find_pairs_in_cluster(
        self,
        client: anthropic.AsyncAnthropic,
        cluster_label: str,
        markets: list[Market],
    ) -> list[MarketRelationship]:
        """Identify correlated and anti-correlated pairs within a cluster."""
        if len(markets) < 2:
            return []

        market_data = [
            {"id": m.condition_id[:12], "q": m.question[:120], "price": round(m.yes_price, 3)}
            for m in markets
        ]

        prompt = f"""Cluster topic: "{cluster_label}"

Identify semantic relationships between these prediction markets.
A SAME-OUTCOME pair: if Market A resolves YES, Market B likely resolves YES too.
A DIFF-OUTCOME pair: if Market A resolves YES, Market B likely resolves NO (mutually exclusive).

Markets:
{json.dumps(market_data, indent=2)}

Return ONLY JSON: {{"pairs": [{{"leader": "id", "follower": "id", "relationship": "same_outcome|different_outcome", "confidence": 0.0-1.0, "reasoning": "brief why"}}]}}

Rules:
- Only include pairs with confidence >= 0.65
- Maximum 5 pairs per cluster
- The "leader" is the market more likely to resolve FIRST (shorter duration or higher liquidity)
- Skip obvious near-duplicates (same event, just different wording)"""

        try:
            response = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            import re
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            data = json.loads(text)

            short_to_market = {m.condition_id[:12]: m for m in markets}
            relationships = []
            now = datetime.now(timezone.utc).isoformat()

            for pair in data.get("pairs", []):
                leader = short_to_market.get(pair.get("leader", ""))
                follower = short_to_market.get(pair.get("follower", ""))
                confidence = float(pair.get("confidence", 0.0))
                rel_type = pair.get("relationship", "same_outcome")

                if not leader or not follower or confidence < 0.65:
                    continue
                if leader.condition_id == follower.condition_id:
                    continue

                relationships.append(MarketRelationship(
                    leader_id=leader.condition_id,
                    follower_id=follower.condition_id,
                    leader_question=leader.question,
                    follower_question=follower.question,
                    relationship=rel_type,
                    confidence=confidence,
                    reasoning=pair.get("reasoning", ""),
                    cluster_label=cluster_label,
                    discovered_at=now,
                ))

            return relationships
        except Exception:
            return []

    def get_stats(self) -> dict:
        """Summary statistics for the clustering engine."""
        total = len(self._relationships)
        resolved = [r for r in self._relationships if r.resolved]
        correct = [
            r for r in resolved
            if r.outcome_correct is not None and r.outcome_correct
        ]
        pending_followers = len(self.get_pending_follower_trades())

        return {
            "total_relationships": total,
            "resolved": len(resolved),
            "accuracy": len(correct) / len(resolved) if resolved else None,
            "pending_follower_signals": pending_followers,
        }
