"""
Astra Alerting — rate-limited webhook notifications.

Sends alerts via HTTP POST to a webhook URL (Slack/Discord compatible).
Rate-limited per alert_type to prevent alert storms.
Never raises exceptions — fire-and-forget with logging.
"""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

logger = logging.getLogger("astra.alerts")


class AlertManager:
    """Rate-limited alert sender."""

    def __init__(
        self,
        webhook_url: str = "",
        cooldown_seconds: int = 300,
        now_fn: Optional[Callable[[], float]] = None,
    ):
        self.webhook_url = webhook_url
        self.cooldown_seconds = cooldown_seconds
        self._now_fn = now_fn or time.monotonic
        self._last_sent: dict[str, float] = {}  # alert_type -> monotonic time

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        alert_type: str = "general",
    ) -> bool:
        """
        Send alert if not rate-limited.

        Args:
            title: Alert title
            message: Alert body
            severity: critical | warning | info
            alert_type: Rate-limit key (e.g., "feed_disconnect", "daily_summary")

        Returns:
            True if alert was sent (or would have been sent if no webhook), False if rate-limited
        """
        # Rate limiting check
        now = self._now_fn()
        last = self._last_sent.get(alert_type, -float("inf"))  # First alert always allowed
        if now - last < self.cooldown_seconds:
            logger.debug(f"Alert rate-limited: {alert_type} (cooldown {self.cooldown_seconds}s)")
            return False

        self._last_sent[alert_type] = now

        # Always log the alert
        log_level = {"critical": logging.ERROR, "warning": logging.WARNING}.get(severity, logging.INFO)
        logger.log(log_level, f"ALERT [{severity}] {title}: {message}")

        # Send to webhook if configured
        if not self.webhook_url:
            return True  # No webhook, but alert was "sent" (logged)

        try:
            payload = json.dumps(
                {
                    "text": f"[{severity.upper()}] *{title}*\n{message}",
                    "severity": severity,
                    "alert_type": alert_type,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception as e:
            logger.warning(f"Alert webhook failed (non-fatal): {e}")
            return True  # Alert was logged even if webhook failed
