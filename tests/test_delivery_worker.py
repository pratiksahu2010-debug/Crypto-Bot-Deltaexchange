"""Telegram outbox worker delivery and retry tests."""
import asyncio
from datetime import datetime, timedelta, timezone

from tgbot.delivery_worker import TelegramDeliveryWorker


class FakeTelegram:
    def __init__(self, fail=False):
        self.bot = object()
        self.fail = fail
        self.sent = []

    async def send_signal_to_chat(self, signal, chat_id):
        if self.fail:
            raise RuntimeError("temporary Telegram outage")
        self.sent.append((signal, chat_id))


def test_delivery_worker_sends_each_claimed_item_to_its_recipient(monkeypatch):
    from database import repositories

    async def run():
        expiry = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
        items = [{
            "delivery_key": "BTC:normal:LONG:1:123",
            "chat_id": 123,
            "signal": {"pair": "B-BTC_USDT", "direction": "LONG"},
            "expires_at": expiry,
            "attempts": 1,
        }]
        completed = []
        monkeypatch.setattr(repositories, "expire_stale_signal_deliveries", lambda: _true())
        monkeypatch.setattr(repositories, "claim_due_signal_deliveries", lambda limit, lease_seconds=90: _items(items))
        monkeypatch.setattr(repositories, "complete_signal_delivery", lambda key, attempts=None: _complete(key, completed))
        telegram = FakeTelegram()
        worker = TelegramDeliveryWorker(telegram)
        assert await worker.run_once() == 1
        assert telegram.sent[0][1] == 123
        assert completed == [items[0]["delivery_key"]]

    async def _true():
        return 0

    async def _items(items):
        return items

    async def _complete(key, completed):
        completed.append(key)
        return True

    asyncio.run(run())


def test_delivery_worker_retries_failed_recipient(monkeypatch):
    from database import repositories

    async def run():
        expiry = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
        items = [{
            "delivery_key": "ETH:momentum:SHORT:2:456",
            "chat_id": 456,
            "signal": {"pair": "B-ETH_USDT", "direction": "SHORT"},
            "expires_at": expiry,
            "attempts": 2,
        }]
        retries = []

        async def fail(key, error, attempts):
            retries.append((key, error, attempts))
            return True

        async def no_expiry():
            return 0

        async def claim(limit, lease_seconds=90):
            return items

        monkeypatch.setattr(repositories, "expire_stale_signal_deliveries", no_expiry)
        monkeypatch.setattr(repositories, "claim_due_signal_deliveries", claim)
        monkeypatch.setattr(repositories, "retry_signal_delivery", fail)
        telegram = FakeTelegram(fail=True)
        worker = TelegramDeliveryWorker(telegram)
        assert await worker.run_once() == 0
        assert len(retries) == 1
        assert retries[0][0] == items[0]["delivery_key"]
        assert retries[0][2] == 2

    asyncio.run(run())
