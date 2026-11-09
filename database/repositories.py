"""Simple repository helpers (optional persistence)."""
from typing import Optional
from utils.logger import logger
from database import db as database_db  # public module alias for test/runtime adapters


async def save_signal(signal: dict) -> bool:
    """Persist an active delivered signal for restart recovery and outcome tracking."""
    import json
    from database import db as database_db
    from database.models import SignalRecord

    factory = database_db.SessionLocal
    if factory is None:
        logger.error("Cannot persist signal: database session unavailable")
        return False
    async with factory() as session:
        try:
            rec = SignalRecord(
                pair=signal["pair"],
                engine=signal["engine"],
                direction=signal["direction"],
                score=signal.get("score", 0),
                grade=signal.get("grade", ""),
                price=float(signal.get("price", 0)),
                status=signal.get("status", "active"),
                payload=json.dumps(signal, default=str),
            )
            session.add(rec)
            await session.commit()
            return True
        except Exception as exc:
            await session.rollback()
            logger.error(f"Signal persistence failed: {exc}")
            return False


async def load_active_signals(max_age_hours: int = 4) -> list:
    """Load unexpired active signal payloads for in-memory lifecycle restoration."""
    import json
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import select, update
    from database import db as database_db
    from database.models import SignalRecord

    factory = database_db.SessionLocal
    if factory is None:
        return []
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=max(1, int(max_age_hours)))
    async with factory() as session:
        try:
            # Mark orphaned active rows older than the configured signal TTL.
            await session.execute(
                update(SignalRecord)
                .where(SignalRecord.status == "active", SignalRecord.created_at < cutoff)
                .values(status="expired")
            )
            result = await session.execute(
                select(SignalRecord)
                .where(SignalRecord.status == "active", SignalRecord.created_at >= cutoff)
                .order_by(SignalRecord.created_at.desc())
            )
            records = result.scalars().all()
            signals = []
            now_utc = datetime.now(timezone.utc)
            for record in records:
                try:
                    signal = json.loads(record.payload or "{}")
                    expires_at = datetime.fromisoformat(
                        str(signal.get("expires_at", "")).replace("Z", "+00:00")
                    )
                    if expires_at <= now_utc:
                        record.status = "expired"
                        continue
                    if signal.get("pair") and signal.get("engine") and signal.get("direction"):
                        signals.append(signal)
                    else:
                        record.status = "invalid"
                except (TypeError, ValueError):
                    record.status = "invalid"
                    logger.warning(f"Skipping malformed persisted signal row id={record.id}")
            await session.commit()
            return signals
        except Exception as exc:
            await session.rollback()
            logger.error(f"Active signal restore query failed: {exc}")
            return []


async def update_signal_record(signal: dict, status: str) -> bool:
    """Persist TP/SL/expiry lifecycle changes for matching active signal records."""
    import json
    from sqlalchemy import select
    from database import db as database_db
    from database.models import SignalRecord

    factory = database_db.SessionLocal
    if factory is None:
        return False
    async with factory() as session:
        try:
            result = await session.execute(
                select(SignalRecord).where(
                    SignalRecord.pair == signal["pair"],
                    SignalRecord.engine == signal["engine"],
                    SignalRecord.direction == signal["direction"],
                    SignalRecord.status == "active",
                )
            )
            records = result.scalars().all()
            for record in records:
                record.status = status
                record.payload = json.dumps(signal, default=str)
            await session.commit()
            return bool(records)
        except Exception as exc:
            await session.rollback()
            logger.error(f"Signal lifecycle persistence failed: {exc}")
            return False


async def claim_signal_once(signal: dict, candle_time_ms: int, ttl_hours: int = 4) -> bool:
    """Atomically claim a signal; the unique key arbitrates simultaneous inserts."""
    import json
    from datetime import datetime, timedelta, timezone
    from sqlalchemy.dialects.postgresql import insert
    from database import db as database_db
    from database.models import SignalDedupRecord

    factory = database_db.SessionLocal
    if factory is None:
        logger.error("Persistent signal deduplication unavailable; suppressing alert")
        return False
    key = f"{signal['pair']}:{signal['direction']}:{int(candle_time_ms)}"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    expires = now + timedelta(hours=max(1, int(ttl_hours)))
    values = dict(dedup_key=key, candle_time_ms=int(candle_time_ms),
                  payload=json.dumps(signal, default=str), created_at=now, expires_at=expires)
    async with factory() as session:
        try:
            # Prune expired claims in the same transaction as the new claim.
            # This preserves the existing retention contract while the unique
            # constraint + ON CONFLICT statement still arbitrates concurrent claims.
            from sqlalchemy import delete
            await session.execute(
                delete(SignalDedupRecord).where(SignalDedupRecord.expires_at <= now)
            )
            statement = insert(SignalDedupRecord).values(**values)
            statement = statement.on_conflict_do_update(
                index_elements=[SignalDedupRecord.dedup_key],
                set_={"candle_time_ms": statement.excluded.candle_time_ms,
                      "payload": statement.excluded.payload,
                      "created_at": statement.excluded.created_at,
                      "expires_at": statement.excluded.expires_at},
                where=SignalDedupRecord.expires_at <= now,
            ).returning(SignalDedupRecord.dedup_key)
            result = await session.execute(statement)
            claimed = result.scalar_one_or_none() is not None
            await session.commit()
            return claimed
        except Exception as exc:
            await session.rollback()
            logger.error(f"Persistent signal claim failed: {exc}")
            return False


async def save_pending_retest(key: str, pending: dict) -> bool:
    """Atomically upsert a pending retest by its stable primary key."""
    import json
    from datetime import datetime, timezone
    from sqlalchemy.dialects.postgresql import insert
    from database import db as database_db
    from database.models import PendingRetestRecord
    factory = database_db.SessionLocal
    if factory is None:
        return False
    expires = datetime.fromtimestamp(float(pending["expires_at"]), tz=timezone.utc).replace(tzinfo=None)
    signal = pending.get("signal") or {}
    values = dict(key=key, pair=signal.get("pair", ""), engine=signal.get("engine", ""),
                  direction=signal.get("direction", ""), expires_at=expires,
                  payload=json.dumps(pending, default=str), updated_at=datetime.now(timezone.utc).replace(tzinfo=None))
    async with factory() as session:
        try:
            statement = insert(PendingRetestRecord).values(**values)
            statement = statement.on_conflict_do_update(
                index_elements=[PendingRetestRecord.key],
                set_={name: getattr(statement.excluded, name) for name in values if name != "key"},
            )
            await session.execute(statement)
            await session.commit()
            return True
        except Exception as exc:
            await session.rollback()
            logger.error(f"Pending retest persistence failed for {key}: {exc}")
            return False

async def load_pending_retests(limit: int = 500) -> list:
    """Restore only valid, unexpired pending retests."""
    import json
    from datetime import datetime, timezone
    from sqlalchemy import select, delete
    from database import db as database_db
    from database.models import PendingRetestRecord
    factory = database_db.SessionLocal
    if factory is None:
        return []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    async with factory() as session:
        try:
            await session.execute(delete(PendingRetestRecord).where(PendingRetestRecord.expires_at <= now))
            result = await session.execute(select(PendingRetestRecord).where(
                PendingRetestRecord.expires_at > now).order_by(PendingRetestRecord.updated_at.desc()).limit(max(1, min(int(limit), 1000))))
            rows = result.scalars().all()
            restored = []
            for row in rows:
                try:
                    item = json.loads(row.payload or "{}")
                    signal = item.get("signal") or {}
                    if signal.get("pair") == row.pair and signal.get("engine") == row.engine and signal.get("direction") == row.direction:
                        restored.append((row.key, item))
                except (TypeError, ValueError):
                    logger.warning(f"Invalid pending retest payload skipped: {row.key}")
            await session.commit()
            return restored
        except Exception as exc:
            await session.rollback()
            logger.error(f"Pending retest recovery failed: {exc}")
            return []


async def delete_pending_retest(key: str) -> bool:
    from sqlalchemy import delete
    from database import db as database_db
    from database.models import PendingRetestRecord
    factory = database_db.SessionLocal
    if factory is None:
        return False
    async with factory() as session:
        try:
            await session.execute(delete(PendingRetestRecord).where(PendingRetestRecord.key == key))
            await session.commit()
            return True
        except Exception as exc:
            await session.rollback()
            logger.error(f"Pending retest delete failed for {key}: {exc}")
            return False


async def record_signal_outcome(signal: dict, latest_price: float, outcome: str = "ACTIVE") -> bool:
    """Persist MFE/MAE and lifecycle outcome for one delivered signal."""
    import json
    from datetime import datetime, timezone
    from sqlalchemy import select
    from database import db as database_db
    from database.models import SignalOutcomeRecord
    factory = database_db.SessionLocal
    if factory is None:
        return False
    try:
        entry = float(signal.get("live_price") or signal.get("price") or 0)
        latest_price = float(latest_price)
    except (TypeError, ValueError, OverflowError):
        return False
    direction = signal.get("direction")
    if (
        direction not in ("LONG", "SHORT")
        or not all(__import__("math").isfinite(value) and value > 0 for value in (entry, latest_price))
    ):
        return False
    pair, engine = signal.get("pair"), signal.get("engine")
    candle_key = signal.get("candle_time_ms") or signal.get("timestamp")
    if not pair or not engine or candle_key is None:
        return False
    allowed_outcomes = {"ACTIVE", "TP1", "TP2", "TP3", "STOP", "FAILED_BREAKOUT"}
    outcome = str(outcome).upper()
    if outcome not in allowed_outcomes:
        return False
    move = ((latest_price / entry) - 1.0) * 100.0 * (1 if direction == "LONG" else -1)
    key = f"{pair}:{engine}:{direction}:{candle_key}"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    async with factory() as session:
        try:
            row = await session.get(SignalOutcomeRecord, None) if False else None
            result = await session.execute(select(SignalOutcomeRecord).where(SignalOutcomeRecord.signal_key == key))
            row = result.scalar_one_or_none()
            if row is None:
                row = SignalOutcomeRecord(signal_key=key, pair=signal.get("pair", ""), engine=signal.get("engine", ""),
                    strategy=signal.get("strategy", ""), direction=direction, entry_price=entry,
                    latest_price=latest_price, mfe_pct=max(0.0, move), mae_pct=min(0.0, move),
                    outcome=outcome, payload=json.dumps(signal, default=str), created_at=now, updated_at=now)
                session.add(row)
            else:
                row.latest_price = latest_price
                row.mfe_pct = max(row.mfe_pct, move)
                row.mae_pct = min(row.mae_pct, move)
                # Preserve the furthest target milestone; terminal stop outcomes
                # override partial targets, while later ACTIVE samples never downgrade it.
                priority = {
                    "ACTIVE": 0, "TP1": 1, "TP2": 2, "TP3": 3,
                    "STOP": 4, "FAILED_BREAKOUT": 4,
                }
                if priority[outcome] >= priority.get(row.outcome, 0):
                    row.outcome = outcome
                row.payload = json.dumps(signal, default=str)
                row.updated_at = now
            await session.commit()
            return True
        except Exception as exc:
            await session.rollback()
            logger.error(f"Signal outcome persistence failed: {exc}")
            return False


async def signal_performance_report(days: int = 30) -> dict:
    """Summarize durable outcomes by engine and strategy."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import select
    from collections import defaultdict
    from database import db as database_db
    from database.models import SignalOutcomeRecord
    factory = database_db.SessionLocal
    if factory is None:
        return {"available": False, "groups": {}}
    async with factory() as session:
        result = await session.execute(select(SignalOutcomeRecord).where(
            SignalOutcomeRecord.created_at >= datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=max(1, min(days, 365)))))
        rows = result.scalars().all()
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row.engine}:{row.strategy}"].append(row)
    report = {}
    for name, items in groups.items():
        ages = []
        payloads = {}
        import json
        for item in items:
            try:
                payload = json.loads(item.payload or "{}")
                payloads[item.id] = payload if isinstance(payload, dict) else {}
                ages.append(float(payloads[item.id].get("signal_age_seconds", 0)))
            except (TypeError, ValueError):
                payloads[item.id] = {}
        failed_breakouts = sum(x.outcome == "FAILED_BREAKOUT" for x in items)
        target_rank = {"TP1": 1, "TP2": 2, "TP3": 3}
        def reached(item, level):
            payload = payloads.get(item.id, {})
            # Persisted target flags preserve partial wins even if the trade
            # later exits at its stop; older rows fall back to final outcome.
            return (
                target_rank.get(item.outcome, 0) >= level
                or any(payload.get(f"tp{n}_hit") is True for n in range(level, 4))
            )
        report[name] = {
            "count": len(items),
            "tp1_rate": sum(reached(x, 1) for x in items) / len(items),
            "tp2_rate": sum(reached(x, 2) for x in items) / len(items),
            "tp3_rate": sum(reached(x, 3) for x in items) / len(items),
            "stop_rate": sum(x.outcome in ("STOP", "FAILED_BREAKOUT") for x in items) / len(items),
            "false_breakout_rate": failed_breakouts / len(items),
            "avg_mfe_pct": sum(x.mfe_pct for x in items) / len(items),
            "avg_mae_pct": sum(x.mae_pct for x in items) / len(items),
            "avg_signal_age_seconds": sum(ages) / len(ages) if ages else None,
            "active": sum(x.outcome == "ACTIVE" for x in items),
        }
    return {"available": True, "groups": report}


async def release_signal_claim(signal: dict, candle_time_ms: int) -> bool:
    """Release an un-delivered dedup claim so a failed notification can retry."""
    from sqlalchemy import delete
    from database import db as database_db
    from database.models import SignalDedupRecord
    factory = database_db.SessionLocal
    if factory is None:
        return False
    key = f"{signal['pair']}:{signal['direction']}:{int(candle_time_ms)}"
    async with factory() as session:
        try:
            await session.execute(delete(SignalDedupRecord).where(SignalDedupRecord.dedup_key == key))
            await session.commit()
            return True
        except Exception as exc:
            await session.rollback()
            logger.error(f"Signal claim release failed for {key}: {exc}")
            return False


async def enqueue_signal_deliveries(signal: dict, recipient_ids: list[int]) -> int:
    """Persist one independent Telegram outbox item per recipient."""
    import json
    from datetime import datetime, timedelta, timezone
    from database import db as database_db
    from database.models import SignalDeliveryRecord
    from config import settings

    factory = database_db.SessionLocal
    engine = database_db.engine
    if factory is None or engine is None or not recipient_ids:
        return 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    # A confirmed retest is a fresh live-price event even though its
    # originating candle may be several minutes old. Give it a fresh delivery
    # window; ordinary candle-close signals consume their remaining age budget.
    if signal.get("entry_type") == "RETEST_CONFIRMED":
        remaining = int(settings.max_signal_age_seconds)
    else:
        remaining = int(
            settings.max_signal_age_seconds - float(signal.get("signal_age_seconds", 0))
        )
    if remaining <= 0:
        logger.warning(
            "Telegram outbox rejected stale signal pair=%s age_seconds=%s",
            signal.get("pair"), signal.get("signal_age_seconds"),
        )
        return 0
    next_attempt = now
    key_base = ":".join(str(signal.get(k, "")) for k in (
        "pair", "engine", "direction", "candle_time_ms"
    ))
    dialect = engine.dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        logger.error("Telegram outbox unsupported database dialect=%s", dialect)
        return 0
    values = []
    for recipient in dict.fromkeys(int(x) for x in recipient_ids if int(x) > 0):
        delivery_key = f"{key_base}:{recipient}"
        payload = {"signal": signal, "chat_id": recipient,
                   "expires_at": (now + timedelta(seconds=remaining)).isoformat()}
        values.append({
            "delivery_key": delivery_key,
            "payload": json.dumps(payload, default=str),
            "status": "pending",
            "attempts": 0,
            "next_attempt_at": next_attempt,
            "last_error": "",
            "updated_at": now,
        })
    if not values:
        return 0
    async with factory() as session:
        try:
            statement = insert(SignalDeliveryRecord).values(values).on_conflict_do_nothing(
                index_elements=[SignalDeliveryRecord.delivery_key]
            )
            result = await session.execute(statement)
            await session.commit()
            return int(result.rowcount or 0)
        except Exception as exc:
            await session.rollback()
            logger.error("Telegram outbox enqueue failed: %s", exc)
            return 0


async def claim_due_signal_deliveries(limit: int = 20, lease_seconds: int = 90) -> list[dict]:
    """Atomically lease due outbox rows; stale leases are recovered after crashes."""
    import json
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import select, or_
    from database import db as database_db
    from database.models import SignalDeliveryRecord

    factory = database_db.SessionLocal
    if factory is None:
        return []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    stale_lease = now - timedelta(seconds=max(30, int(lease_seconds)))
    async with factory() as session:
        try:
            # A process crash can leave a row leased in "sending". Recover
            # it after the lease expires, but do not retry indefinitely.
            from sqlalchemy import update
            await session.execute(
                update(SignalDeliveryRecord)
                .where(
                    SignalDeliveryRecord.status == "sending",
                    SignalDeliveryRecord.updated_at <= stale_lease,
                    SignalDeliveryRecord.attempts >= 8,
                )
                .values(
                    status="failed",
                    last_error="maximum delivery attempts exhausted after worker lease expired",
                    updated_at=now,
                )
            )
            query = (
                select(SignalDeliveryRecord)
                .where(or_(
                    ((SignalDeliveryRecord.status.in_(("pending", "retry")))
                     & (SignalDeliveryRecord.next_attempt_at <= now)),
                    ((SignalDeliveryRecord.status == "sending")
                     & (SignalDeliveryRecord.updated_at <= stale_lease)
                     & (SignalDeliveryRecord.attempts < 8)),
                ))
                .order_by(SignalDeliveryRecord.next_attempt_at.asc())
                .limit(max(1, min(int(limit), 100)))
                .with_for_update(skip_locked=True)
            )
            result = await session.execute(query)
            rows = result.scalars().all()
            claimed = []
            for row in rows:
                try:
                    payload = json.loads(row.payload or "{}")
                    if not isinstance(payload, dict) or not payload.get("chat_id") or not isinstance(payload.get("signal"), dict):
                        row.status = "failed"
                        row.last_error = "invalid outbox payload"
                        row.updated_at = now
                        continue
                    row.status = "sending"
                    row.attempts = int(row.attempts or 0) + 1
                    row.updated_at = now
                    claimed.append({
                        "delivery_key": row.delivery_key,
                        "attempts": row.attempts,
                        **payload,
                    })
                except (TypeError, ValueError):
                    row.status = "failed"
                    row.last_error = "malformed outbox JSON"
                    row.updated_at = now
            await session.commit()
            return claimed
        except Exception as exc:
            await session.rollback()
            logger.error("Telegram outbox claim failed: %s", exc)
            return []


async def complete_signal_delivery(delivery_key: str, attempts: int | None = None) -> bool:
    """Mark delivery complete only if this worker still owns the sending attempt."""
    from datetime import datetime, timezone
    from sqlalchemy import update
    from database import db as database_db
    from database.models import SignalDeliveryRecord
    factory = database_db.SessionLocal
    if factory is None:
        return False
    async with factory() as session:
        try:
            conditions = [
                SignalDeliveryRecord.delivery_key == delivery_key,
                SignalDeliveryRecord.status == "sending",
            ]
            if attempts is not None:
                conditions.append(SignalDeliveryRecord.attempts == int(attempts))
            result = await session.execute(
                update(SignalDeliveryRecord)
                .where(*conditions)
                .values(
                    status="delivered",
                    last_error="",
                    updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            await session.commit()
            return bool(result.rowcount)
        except Exception as exc:
            await session.rollback()
            logger.error("Telegram outbox completion failed: %s", exc)
            return False


async def retry_signal_delivery(
    delivery_key: str, error: str, attempts: int, max_attempts: int = 8
) -> bool:
    """Schedule retry only while the supplied worker attempt still owns the row."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import update
    from database import db as database_db
    from database.models import SignalDeliveryRecord
    factory = database_db.SessionLocal
    if factory is None:
        return False
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    attempt_number = max(1, int(attempts))
    terminal = attempt_number >= max(1, int(max_attempts))
    delay = min(60, 2 ** min(attempt_number, 6))
    async with factory() as session:
        try:
            result = await session.execute(
                update(SignalDeliveryRecord)
                .where(
                    SignalDeliveryRecord.delivery_key == delivery_key,
                    SignalDeliveryRecord.status == "sending",
                    SignalDeliveryRecord.attempts == attempt_number,
                )
                .values(
                    status="failed" if terminal else "retry",
                    last_error=str(error)[:2000],
                    next_attempt_at=now + timedelta(seconds=delay),
                    updated_at=now,
                )
            )
            await session.commit()
            return bool(result.rowcount)
        except Exception as exc:
            await session.rollback()
            logger.error("Telegram outbox retry update failed: %s", exc)
            return False


async def expire_stale_signal_deliveries(limit: int = 100) -> int:
    """Expire pending items whose original signal validity window has passed."""
    import json
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import select, delete
    from database import db as database_db
    from database.models import SignalDeliveryRecord
    factory = database_db.SessionLocal
    if factory is None:
        return 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    async with factory() as session:
        try:
            # Bound outbox growth by pruning terminal rows after 30 days.
            await session.execute(delete(SignalDeliveryRecord).where(
                SignalDeliveryRecord.status.in_(("delivered", "failed", "expired")),
                SignalDeliveryRecord.updated_at < now - timedelta(days=30),
            ))
            result = await session.execute(select(SignalDeliveryRecord).where(
                SignalDeliveryRecord.status.in_(("pending", "retry"))
            ).limit(max(1, min(int(limit), 500))))
            count = 0
            for row in result.scalars().all():
                try:
                    payload = json.loads(row.payload or "{}")
                    expiry = datetime.fromisoformat(str(payload.get("expires_at", "")))
                    if expiry.tzinfo is not None:
                        expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)
                    if expiry <= now:
                        row.status = "expired"
                        row.last_error = "signal entry window expired before Telegram delivery"
                        row.updated_at = now
                        count += 1
                except (TypeError, ValueError):
                    row.status = "failed"
                    row.last_error = "invalid expiry in outbox payload"
                    row.updated_at = now
            await session.commit()
            return count
        except Exception as exc:
            await session.rollback()
            logger.error("Telegram outbox expiry sweep failed: %s", exc)
            return 0


async def signal_delivery_queue_stats() -> dict:
    """Return durable Telegram outbox counts and age of the oldest live item."""
    from datetime import datetime, timezone
    from sqlalchemy import select, func
    from database import db as database_db
    from database.models import SignalDeliveryRecord

    factory = database_db.SessionLocal
    if factory is None:
        return {"available": False, "counts": {}, "total": 0, "oldest_pending_age_seconds": None}
    statuses = ("pending", "retry", "sending", "delivered", "failed", "expired")
    try:
        async with factory() as session:
            result = await session.execute(
                select(SignalDeliveryRecord.status, func.count(SignalDeliveryRecord.delivery_key))
                .group_by(SignalDeliveryRecord.status)
            )
            counts = {status: 0 for status in statuses}
            counts.update({str(status): int(count) for status, count in result.all()})
            oldest_result = await session.execute(
                select(func.min(func.coalesce(
                    SignalDeliveryRecord.created_at, SignalDeliveryRecord.updated_at
                ))).where(
                    SignalDeliveryRecord.status.in_(("pending", "retry", "sending"))
                )
            )
            oldest = oldest_result.scalar_one_or_none()
        age = None
        if oldest is not None:
            if oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=timezone.utc)
            age = max(0.0, (datetime.now(timezone.utc) - oldest).total_seconds())
        return {
            "available": True, "counts": counts, "total": sum(counts.values()),
            "oldest_pending_age_seconds": round(age, 1) if age is not None else None,
        }
    except Exception as exc:
        logger.error("Telegram outbox stats query failed: %s", exc)
        return {"available": False, "counts": {}, "total": 0, "oldest_pending_age_seconds": None}


async def expire_signal_delivery(delivery_key: str, attempts: int) -> bool:
    """Expire an item only while the supplied worker still owns its lease."""
    from datetime import datetime, timezone
    from sqlalchemy import update
    from database import db as database_db
    from database.models import SignalDeliveryRecord
    factory = database_db.SessionLocal
    if factory is None:
        return False
    async with factory() as session:
        try:
            result = await session.execute(
                update(SignalDeliveryRecord)
                .where(
                    SignalDeliveryRecord.delivery_key == delivery_key,
                    SignalDeliveryRecord.status == "sending",
                    SignalDeliveryRecord.attempts == max(1, int(attempts)),
                )
                .values(
                    status="expired",
                    last_error="signal entry window expired before Telegram delivery",
                    updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            await session.commit()
            return bool(result.rowcount)
        except Exception as exc:
            await session.rollback()
            logger.error("Telegram outbox expiry update failed: %s", exc)
            return False
