"""Async SQLAlchemy database lifecycle with production PostgreSQL enforcement."""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from config import settings
from utils.logger import logger


class Base(DeclarativeBase):
    pass


async def _upgrade_postgres_schema(conn) -> None:
    """Apply idempotent schema fixes not handled by create_all."""
    if conn.dialect.name == "postgresql":
        # Unix timestamps in milliseconds exceed PostgreSQL INTEGER's 32-bit range.
        await conn.execute(text(
            "ALTER TABLE signal_dedup ALTER COLUMN candle_time_ms TYPE BIGINT"
        ))

    # Keep the original outbox enqueue time immutable so queue-age metrics do
    # not reset when a worker claims or retries a delivery. This migration is
    # portable across PostgreSQL and SQLite; legacy rows use their last known
    # update timestamp as the best available approximation.
    tables = await conn.run_sync(
        lambda sync_conn: __import__("sqlalchemy").inspect(sync_conn).get_table_names()
    )
    if "signal_delivery_outbox" in tables:
        columns = await conn.run_sync(
            lambda sync_conn: {
                column["name"] for column in
                __import__("sqlalchemy").inspect(sync_conn).get_columns("signal_delivery_outbox")
            }
        )
        if "created_at" not in columns:
            await conn.execute(text(
                "ALTER TABLE signal_delivery_outbox ADD COLUMN created_at TIMESTAMP"
            ))
            await conn.execute(text(
                "UPDATE signal_delivery_outbox SET created_at = updated_at WHERE created_at IS NULL"
            ))


engine = None
SessionLocal = None


async def init_db() -> None:
    global engine, SessionLocal
    # Import models before create_all so every persistent table is registered.
    from database import models  # noqa: F401
    url = settings.database_url
    # Render's local filesystem is ephemeral on the current service plan.
    # Refuse production scans with SQLite so deduplication and retests survive restarts.
    if settings.is_production and not settings.use_postgres:
        raise RuntimeError(
            "Production requires DATABASE_URL pointing to PostgreSQL; "
            "SQLite is permitted only for local development."
        )
    if settings.use_postgres:
        # Render commonly supplies postgres:// or postgresql:// URLs; use the
        # asyncpg driver already included in requirements.
        if url.startswith("postgres://"):
            url = "postgresql+asyncpg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]
        url = url.replace("sslmode=require", "ssl=require")
        engine = create_async_engine(
            url,
            pool_size=2,
            max_overflow=3,
            pool_pre_ping=True,
            pool_recycle=300,
        )
    else:
        engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            # Give short concurrent writes from the Normal and Momentum engines
            # time to complete instead of failing immediately with SQLITE_BUSY.
            connect_args={"timeout": 30},
        )
        logger.warning(
            "Using SQLite. State and deduplication work while this Render instance "
            "is running, but the Free filesystem is ephemeral and state can reset "
            "after a restart or redeploy."
        )

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
            await conn.run_sync(Base.metadata.create_all)
            await _upgrade_postgres_schema(conn)
            existing = await conn.run_sync(
                lambda sync_conn: __import__("sqlalchemy").inspect(sync_conn).get_table_names()
            )
            required = set(Base.metadata.tables)
            missing = sorted(required - set(existing))
            if missing:
                raise RuntimeError(f"Database schema check failed; missing tables: {missing}")
            await conn.execute(text("DELETE FROM signals WHERE engine = 'signals'"))
            await conn.execute(text("DELETE FROM watchlist WHERE engine = 'signals'"))
            await conn.execute(text("DELETE FROM signal_dedup WHERE dedup_key LIKE 'signals:%'"))
            await conn.execute(text("DROP TABLE IF EXISTS step_channel_state"))
        logger.info("DATABASE_READY dialect=%s connectivity=ok schema=ok tables=%d", engine.dialect.name, len(required))
    except Exception:
        logger.exception("DATABASE_STARTUP_CHECK_FAILED connectivity_or_schema=true")
        await engine.dispose()
        engine = None
        SessionLocal = None
        raise


async def get_session() -> AsyncSession:
    assert SessionLocal is not None
    async with SessionLocal() as session:
        yield session

async def dispose_db() -> None:
    """Close the SQLAlchemy pool cleanly during application shutdown."""
    global engine, SessionLocal
    if engine is not None:
        await engine.dispose()
    engine = None
    SessionLocal = None
