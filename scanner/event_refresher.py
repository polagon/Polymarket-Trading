"""
Event Metadata Refresher (GAP #7 FIX).

CRITICAL: Periodic refresh of event metadata to catch negRisk flag changes.

negRisk events break parity assumptions and must be detected early.
"""

import asyncio
import logging
from typing import Dict, List

import aiohttp

from config import GAMMA_API_URL
from models.types import Event, Market

logger = logging.getLogger(__name__)


class EventRefresher:
    """
    Periodic refresher for event metadata.

    CRITICAL: Detects negRisk flag changes that would invalidate parity trades.
    """

    def __init__(self, refresh_interval_seconds: int = 600):
        """
        Initialize event refresher.

        Args:
            refresh_interval_seconds: How often to refresh (default 10 minutes)
        """
        self.refresh_interval = refresh_interval_seconds
        self.event_cache: Dict[str, Event] = {}
        self.running = False

    async def refresh_events(self, markets: List[Market]) -> Dict[str, Event]:
        """
        Refresh event metadata for all markets.

        CRITICAL: If negRisk flag is added to an event, all markets in that
        event must trigger cancel-all.

        Args:
            markets: List of markets to refresh events for

        Returns:
            Dict of event_id → Event
        """
        event_ids = set()
        for market in markets:
            if market.event:
                event_ids.add(market.event.event_id)

        if not event_ids:
            return {}

        refreshed_events = {}

        async with aiohttp.ClientSession() as session:
            for event_id in event_ids:
                try:
                    async with session.get(
                        f"{GAMMA_API_URL}/events/{event_id}",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            event = Event(
                                event_id=data.get("id", event_id),
                                title=data.get("title", ""),
                                neg_risk=data.get("negRisk", False),
                                augmented_neg_risk=data.get("augmentedNegRisk", False),
                                metadata=data,
                            )
                            refreshed_events[event_id] = event

                            # Check for flag changes
                            if event_id in self.event_cache:
                                old_event = self.event_cache[event_id]
                                if not old_event.neg_risk and event.neg_risk:
                                    logger.error(
                                        f"🚨 negRisk FLAG ADDED: Event {event_id} ({event.title}). "
                                        f"MUST cancel all orders in this event!"
                                    )
                                if not old_event.augmented_neg_risk and event.augmented_neg_risk:
                                    logger.warning(f"⚠️ augmentedNegRisk FLAG ADDED: Event {event_id} ({event.title})")

                except Exception as e:
                    logger.warning(f"Failed to refresh event {event_id}: {e}")

                await asyncio.sleep(0.1)  # Rate limiting

        # Update cache
        self.event_cache.update(refreshed_events)

        logger.info(f"Refreshed {len(refreshed_events)} events")
        return refreshed_events

    def get_negRisk_markets(self, markets: List[Market]) -> List[Market]:
        """
        Get list of markets that are in negRisk events.

        Args:
            markets: List of markets

        Returns:
            List of markets in negRisk events
        """
        negRisk_markets = []
        for market in markets:
            if market.event:
                event = self.event_cache.get(market.event.event_id)
                if event and (event.neg_risk or event.augmented_neg_risk):
                    negRisk_markets.append(market)

        return negRisk_markets

    async def start_periodic_refresh(self, get_markets_callback):
        """
        Start periodic event refresh loop.

        Args:
            get_markets_callback: Async function that returns current market list
        """
        self.running = True
        logger.info(f"Event refresher started (interval={self.refresh_interval}s)")

        while self.running:
            try:
                markets = await get_markets_callback()
                await self.refresh_events(markets)
            except Exception as e:
                logger.error(f"Event refresh failed: {e}", exc_info=True)

            await asyncio.sleep(self.refresh_interval)

    def stop(self):
        """Stop periodic refresh."""
        self.running = False
        logger.info("Event refresher stopped")
