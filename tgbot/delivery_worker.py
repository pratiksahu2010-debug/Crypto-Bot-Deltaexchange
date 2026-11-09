"""Background worker for durable, per-recipient Telegram signal delivery."""
import asyncio
from datetime import datetime, timezone
from typing import Optional

from utils.logger import logger


class TelegramDeliveryWorker:
    def __init__(self, telegram, poll_seconds: float = 1.0, batch_size: int = 20):
        self.telegram = telegram
        self.poll_seconds = max(0.25, float(poll_seconds))
        self.batch_size = max(1, min(int(batch_size), 100))
        self.running = False
        self.delivered = 0
        self.failed = 0
        self.expired = 0

    @staticmethod
    def _expired(item: dict) -> bool:
        try:
            expiry = datetime.fromisoformat(
                str(item.get("expires_at", "")).replace("Z", "+00:00")
            )
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            return expiry <= datetime.now(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return True

    async def run_once(self) -> int:
        """Deliver one due batch; return the number of successful sends."""
        if not self.telegram or not self.telegram.bot:
            return 0
        from database.repositories import (
            claim_due_signal_deliveries,
            complete_signal_delivery,
            retry_signal_delivery,
            expire_signal_delivery,
            expire_stale_signal_deliveries,
        )
        await expire_stale_signal_deliveries()
        # A whole batch is leased before sequential delivery. Size the lease to
        # the worst-case Telegram transport budget so later items are not
        # reclaimed while still waiting their turn in this batch.
        lease_seconds = max(90, self.batch_size * 30 + 30)
        items = await claim_due_signal_deliveries(
            self.batch_size, lease_seconds=lease_seconds
        )
        sent = 0
        for item in items:
            key = item.get("delivery_key", "")
            if self._expired(item):
                await expire_signal_delivery(key, int(item.get("attempts", 1)))
                self.expired += 1
                continue
            try:
                await self.telegram.send_signal_to_chat(
                    item["signal"], int(item["chat_id"])
                )
                if await complete_signal_delivery(key, int(item.get("attempts", 1))):
                    sent += 1
                    self.delivered += 1
                    try:
                        from database.repositories import record_signal_outcome
                        signal = item["signal"]
                        entry_price = float(
                            signal.get("live_price") or signal.get("price") or 0
                        )
                        if entry_price > 0:
                            await record_signal_outcome(signal, entry_price, "ACTIVE")
                    except Exception:
                        logger.exception("Signal outcome recording failed for outbox key=%s", key)
                    logger.info(
                        "TELEGRAM_OUTBOX_DELIVERED key=%s chat_id=%s attempt=%s",
                        key, item["chat_id"], item.get("attempts", 0),
                    )
            except Exception as exc:
                self.failed += 1
                await retry_signal_delivery(
                    key, f"{type(exc).__name__}: {exc}",
                    int(item.get("attempts", 1)),
                )
                logger.warning(
                    "TELEGRAM_OUTBOX_RETRY key=%s chat_id=%s attempt=%s error=%s",
                    key, item.get("chat_id"), item.get("attempts", 0), exc,
                )
        return sent

    async def run_forever(self) -> None:
        self.running = True
        logger.info("TELEGRAM_OUTBOX_WORKER_STARTED poll_seconds=%.2f", self.poll_seconds)
        try:
            while self.running:
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Telegram outbox worker iteration failed")
                await asyncio.sleep(self.poll_seconds)
        finally:
            self.running = False
            logger.info(
                "TELEGRAM_OUTBOX_WORKER_STOPPED delivered=%s failed=%s expired=%s",
                self.delivered, self.failed, self.expired,
            )

    def stop(self) -> None:
        self.running = False
