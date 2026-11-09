"""SQLAlchemy models for persistent state."""
from datetime import datetime, timezone
from sqlalchemy import String, Float, Boolean, DateTime, Text, Integer, BigInteger
from sqlalchemy.orm import Mapped, mapped_column
from database.db import Base


def _utc_naive_now() -> datetime:
    """UTC timestamp stored as naive for the existing SQLAlchemy schema."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class SignalRecord(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String(32), index=True)
    engine: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(8))
    score: Mapped[float] = mapped_column(Float)
    grade: Mapped[str] = mapped_column(String(32))
    price: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="active")
    payload: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)


class WatchlistRecord(Base):
    __tablename__ = "watchlist"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String(32), index=True)
    engine: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(8))
    score: Mapped[float] = mapped_column(Float)
    payload: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)


class SignalDedupRecord(Base):
    """Durable per-pair/direction/candle alert lock shared across both engines."""
    __tablename__ = "signal_dedup"
    dedup_key: Mapped[str] = mapped_column(String(96), primary_key=True)
    candle_time_ms: Mapped[int] = mapped_column(BigInteger, default=0)
    payload: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class PendingRetestRecord(Base):
    """Durable short-lived candidate awaiting a fresh-price retest."""
    __tablename__ = "pending_retests"
    key: Mapped[str] = mapped_column(String(160), primary_key=True)
    pair: Mapped[str] = mapped_column(String(32), index=True)
    engine: Mapped[str] = mapped_column(String(16), index=True)
    direction: Mapped[str] = mapped_column(String(8))
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    payload: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)


class SignalOutcomeRecord(Base):
    """Durable signal outcome and excursion measurements."""
    __tablename__ = "signal_outcomes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_key: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    pair: Mapped[str] = mapped_column(String(32), index=True)
    engine: Mapped[str] = mapped_column(String(16), index=True)
    strategy: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(8))
    entry_price: Mapped[float] = mapped_column(Float)
    latest_price: Mapped[float] = mapped_column(Float)
    mfe_pct: Mapped[float] = mapped_column(Float, default=0.0)
    mae_pct: Mapped[float] = mapped_column(Float, default=0.0)
    outcome: Mapped[str] = mapped_column(String(24), default="ACTIVE", index=True)
    payload: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)


class SignalDeliveryRecord(Base):
    """Durable outbox state for retry-safe Telegram delivery."""
    __tablename__ = "signal_delivery_outbox"
    delivery_key: Mapped[str] = mapped_column(String(160), primary_key=True)
    payload: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now, nullable=True, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)
