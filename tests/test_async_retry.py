import asyncio
import pytest

from utils.retry import async_retry


@pytest.mark.asyncio
async def test_async_retry_adds_bounded_jitter_and_exponential_backoff(monkeypatch):
    waits = []
    attempts = {"count": 0}

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr("utils.retry.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("utils.retry.random.uniform", lambda low, high: high)

    @async_retry(max_attempts=3, delay=1.0, backoff=2.0, jitter=0.25)
    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("temporary exchange timeout")
        return "ok"

    assert await flaky() == "ok"
    assert attempts["count"] == 3
    assert waits == [1.25, 2.5]


@pytest.mark.asyncio
async def test_async_retry_raises_last_exception_after_attempt_limit(monkeypatch):
    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr("utils.retry.asyncio.sleep", fake_sleep)
    calls = {"count": 0}

    @async_retry(max_attempts=2, delay=0, jitter=0)
    async def always_fails():
        calls["count"] += 1
        raise ConnectionError("offline")

    with pytest.raises(ConnectionError, match="offline"):
        await always_fails()
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_async_retry_does_not_swallow_task_cancellation():
    started = asyncio.Event()

    @async_retry(max_attempts=3, delay=0)
    async def cancellable():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(cancellable())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("kwargs", [
    {"max_attempts": 0},
    {"delay": -0.1},
    {"backoff": 0.5},
    {"jitter": -0.1},
    {"jitter": 1.1},
])
def test_async_retry_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        async_retry(**kwargs)


@pytest.mark.asyncio
async def test_async_retry_fails_fast_when_policy_rejects_exception(monkeypatch):
    calls = {"count": 0}
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr("utils.retry.asyncio.sleep", fake_sleep)

    @async_retry(
        max_attempts=3, delay=1, jitter=0,
        retry_if=lambda exc: not isinstance(exc, ValueError),
    )
    async def permanent_failure():
        calls["count"] += 1
        raise ValueError("permanent client error")

    with pytest.raises(ValueError, match="permanent client error"):
        await permanent_failure()
    assert calls["count"] == 1
    assert waits == []


@pytest.mark.asyncio
async def test_async_retry_policy_allows_transient_exception(monkeypatch):
    calls = {"count": 0}

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr("utils.retry.asyncio.sleep", fake_sleep)

    @async_retry(
        max_attempts=2, delay=0, jitter=0,
        retry_if=lambda exc: isinstance(exc, ConnectionError),
    )
    async def transient_failure():
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectionError("temporary network failure")
        return "recovered"

    assert await transient_failure() == "recovered"
    assert calls["count"] == 2
