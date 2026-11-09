"""Simple async retry decorator with bounded exponential jitter."""
import asyncio
import functools
import random
from typing import Callable, Optional, TypeVar, Any

from utils.logger import logger

T = TypeVar("T")


def async_retry(
    max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
    jitter: float = 0.2, retry_if: Optional[Callable[[Exception], bool]] = None,
):
    """Retry selected operation failures with exponential jitter.

    When supplied, retry_if decides whether an exception is transient. Exceptions
    rejected by the predicate are raised immediately; task cancellation propagates.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if delay < 0:
        raise ValueError("delay cannot be negative")
    if backoff < 1:
        raise ValueError("backoff must be at least 1")
    if not 0 <= jitter <= 1:
        raise ValueError("jitter must be between 0 and 1")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if retry_if is not None and not retry_if(exc):
                        raise
                    if attempt == max_attempts:
                        break
                    wait_seconds = current_delay + random.uniform(0.0, current_delay * jitter)
                    logger.warning(
                        f"{func.__name__} attempt {attempt} failed: {exc}. "
                        f"Retrying in {wait_seconds:.2f}s"
                    )
                    await asyncio.sleep(wait_seconds)
                    current_delay *= backoff
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
