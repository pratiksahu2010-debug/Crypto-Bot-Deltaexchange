"""Live (intrabar) Momentum Explosion + ORB ignition.

Why this exists
---------------
The closed-candle scan can only see a move after its 5m candle has closed, so a
big move is on average 2.5 minutes old (worst case 5) before any gate runs, and
the closed-candle gates then ask for even more (see README of the change):

* EXPLOSION needs ATR(14) to expand >= 8% in ONE candle, which takes a candle
  of roughly 3.7x the prior ATR, while also capping EMA9 extension at 1.35 ATR.
* ORB_BREAKOUT rejects the breakout candle as EXTENDED_BREAKOUT whenever it
  closes > 0.8 ATR beyond the range - i.e. exactly the explosive ones.

This module judges the FORMING candle once per second instead:

* measures the move in units of the ATR of COMPLETED candles (so the candle's
  own range cannot inflate the yardstick),
* normalises volume by elapsed time ("pace") instead of waiting for a full
  candle of volume,
* triggers while the move is still near its origin (max-move cap = anti-chase),
* for ORB triggers on the first confirmed break of the opening range, not on
  the first candle CLOSE beyond it.

Everything here is pure: no I/O, no clock reads (``now_ms`` is passed in), so it
is deterministic and unit-testable. All thresholds are in ``LiveConfig``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.scoring_engine import ScoreComponent, ScoreResult
from indicators import atr, bollinger_bands, ema, keltner_channels
from strategies.momentum import orb_detector


@dataclass(frozen=True)
class LiveConfig:
    # --- Explosion ---------------------------------------------------------
    min_move_atr: float = 1.20        # directional move from candle open, in prior-ATR units
    max_move_atr: float = 2.20        # above this the move has "already happened": skip (anti-chase)
    min_elapsed_seconds: float = 10.0
    max_elapsed_fraction: float = 0.85  # later than this the closed-candle path is nearly as fast
    min_volume_pace: float = 1.6      # (candle volume / avg volume) / elapsed fraction
    min_volume_floor: float = 0.35    # and at least this share of an average candle already traded
    max_pre_extension_atr: float = 2.5  # candle open already this far from EMA21 => run is old
    min_atr_pct: float = 0.05         # ignore dead markets (ATR below 0.05% of price)
    impulse_pace: float = 2.5         # evidence: very fast volume + big move even without structure
    impulse_move_atr: float = 1.3
    min_structure_volume_pace: float = 1.8  # structural break must have above-baseline participation
    min_squeeze_volume_pace: float = 2.0    # squeeze release needs stronger participation
    max_squeeze_age_candles: int = 2          # compression must be recent, not stale
    # --- ORB ---------------------------------------------------------------
    orb_buffer_atr: float = 0.15      # a break must clear the edge by this much (not a poke)
    orb_max_extension_atr: float = 0.60
    orb_hold_seconds: float = 8.0     # must stay beyond edge+buffer this long
    orb_min_volume_pace: float = 1.2
    orb_min_volume_floor: float = 0.25
    orb_active_minutes: int = 180
    # --- Shared ------------------------------------------------------------
    min_score: float = 7.0
    sessions: Tuple[str, ...] = ("london", "new_york", "asia")

    @classmethod
    def from_settings(cls, settings: Any) -> "LiveConfig":
        defaults = cls()
        sessions: List[str] = []
        for name, flag in (("london", "orb_london_enabled"), ("new_york", "orb_new_york_enabled"),
                           ("asia", "orb_asia_enabled")):
            if getattr(settings, flag, True):
                sessions.append(name)
        if not getattr(settings, "orb_enabled", True) or not getattr(settings, "live_orb_enabled", True):
            sessions = []

        def pick(attr: str, default: float) -> float:
            try:
                value = float(getattr(settings, attr, default))
                return value if math.isfinite(value) else default
            except (TypeError, ValueError):
                return default

        return cls(
            min_move_atr=pick("live_min_move_atr", defaults.min_move_atr),
            max_move_atr=pick("live_max_move_atr", defaults.max_move_atr),
            min_volume_pace=pick("live_min_volume_pace", defaults.min_volume_pace),
            orb_hold_seconds=pick("live_orb_hold_seconds", defaults.orb_hold_seconds),
            orb_max_extension_atr=pick("live_orb_max_extension_atr", defaults.orb_max_extension_atr),
            orb_active_minutes=int(pick("orb_active_minutes", defaults.orb_active_minutes)),
            min_score=max(8.0, pick("live_min_score", defaults.min_score)),
            sessions=tuple(sessions),
        )


@dataclass
class LiveContext:
    """Per-pair, per-forming-candle facts computed ONCE from completed candles."""
    pair: str
    forming_open_ms: int
    interval_ms: int
    atr: float
    avg_volume: float
    ema9: Optional[float]
    ema21: Optional[float]
    ema50: Optional[float]
    resistance: float
    support: float
    compression_recent: bool
    last_close: float
    compression_age: Optional[int] = None
    orb: Dict[str, dict] = field(default_factory=dict)

    @property
    def atr_pct(self) -> float:
        return self.atr / self.last_close * 100.0 if self.last_close > 0 else 0.0


def _compression_age(high: np.ndarray, low: np.ndarray, close: np.ndarray, bars: int = 6) -> Optional[int]:
    """Return candles since most recent BB-inside-KC compression; None if absent."""
    n = len(close)
    for back in range(bars):
        end = n - back
        if end < 40:
            break
        lo = max(0, end - 80)
        bb = bollinger_bands(close[lo:end], 20, 2.0)
        kc = keltner_channels(high[lo:end], low[lo:end], close[lo:end], 20, 1.5)
        if bb is not None and kc is not None and bb[0] < kc[0] and bb[2] > kc[2]:
            return back
    return None


def _compression_recent(high: np.ndarray, low: np.ndarray, close: np.ndarray, bars: int = 6) -> bool:
    return _compression_age(high, low, close, bars) is not None


def build_context(
    pair: str, arrays: Dict[str, Any], forming_open_ms: int, interval_ms: int,
    now_ms: int, sessions: Tuple[str, ...] = (), orb_active_minutes: int = 180,
) -> Optional[LiveContext]:
    """Build the context from COMPLETED candles; None when data is unusable.

    Requires the candle immediately before the forming one to be present: a gap
    would make the 20-candle structure levels silently wrong.
    """
    try:
        times = np.asarray(arrays["time"], dtype=np.int64)
        done = times < int(forming_open_ms)
        n = int(np.count_nonzero(done))
        if n < 40 or int(times[done][-1]) != int(forming_open_ms) - int(interval_ms):
            return None
        o, h, l, c, v = (np.asarray(arrays[k], dtype=float)[done] for k in
                         ("open", "high", "low", "close", "volume"))
        tail = slice(max(0, n - 150), n)
        atr_v = atr(h[tail], l[tail], c[tail], 14)
        avg_volume = float(np.mean(v[-20:]))
        if atr_v is None or atr_v <= 0 or not math.isfinite(avg_volume) or avg_volume <= 0:
            return None
        ctx = LiveContext(
            pair=pair, forming_open_ms=int(forming_open_ms), interval_ms=int(interval_ms),
            atr=float(atr_v), avg_volume=avg_volume,
            ema9=ema(c[tail], 9), ema21=ema(c[tail], 21), ema50=ema(c[tail], 50),
            resistance=float(np.max(h[-20:])), support=float(np.min(l[-20:])),
            compression_recent=_compression_recent(h[tail], l[tail], c[tail]),
            last_close=float(c[-1]),
            compression_age=_compression_age(h[tail], l[tail], c[tail]),
        )
        if sessions:
            completed = {"time": times[done], "high": h, "low": l}
            for session in sessions:
                rng = orb_detector.opening_range(completed, session, now_ms, orb_active_minutes)
                if rng.get("state") != "READY":
                    continue
                width_atr = (rng["range_high"] - rng["range_low"]) / ctx.atr
                if 0.10 <= width_atr <= 2.50:
                    rng["range_width_atr"] = round(width_atr, 4)
                    ctx.orb[session] = rng
        return ctx
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _candle(candle: Dict[str, Any]) -> Optional[Tuple[float, float, float, float, float, int]]:
    try:
        o, h, l, c = (float(candle[k]) for k in ("open", "high", "low", "close"))
        v = float(candle.get("volume", 0.0) or 0.0)
        t = int(candle["time"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(x) for x in (o, h, l, c, v)) or min(o, h, l, c) <= 0 or h < l or v < 0:
        return None
    return o, h, l, c, v, t


def _trend_points(ctx: LiveContext, direction: str, price: float) -> float:
    pts = 0.0
    if ctx.ema9 is not None and ctx.ema21 is not None:
        pts += 0.5 if ((ctx.ema9 > ctx.ema21) == (direction == "LONG")) else 0.0
    if ctx.ema50 is not None:
        pts += 0.5 if ((price > ctx.ema50) == (direction == "LONG")) else 0.0
    return pts


def _reject(reason: str, **extra: Any) -> Dict[str, Any]:
    return {"state": "REJECT", "reason": reason, **extra}


def evaluate_explosion(ctx: LiveContext, candle: Dict[str, Any], now_ms: int,
                       cfg: LiveConfig = LiveConfig()) -> Dict[str, Any]:
    """Judge the forming candle as a momentum explosion at THIS instant."""
    parsed = _candle(candle)
    if parsed is None:
        return _reject("invalid_candle")
    o, h, l, c, v, t = parsed
    if t != ctx.forming_open_ms:
        return _reject("context_mismatch")
    elapsed_ms = now_ms - t
    frac = elapsed_ms / ctx.interval_ms
    if elapsed_ms < cfg.min_elapsed_seconds * 1000 or frac > cfg.max_elapsed_fraction:
        return _reject("outside_live_window", elapsed_seconds=round(elapsed_ms / 1000, 1))
    if ctx.atr_pct < cfg.min_atr_pct:
        return _reject("atr_too_small")
    # Direction comes from completed-candle EMA alignment, never candle colour.
    if ctx.ema9 is None or ctx.ema21 is None or ctx.ema9 == ctx.ema21:
        return _reject("indicator_direction_unavailable")
    direction = "LONG" if ctx.ema9 > ctx.ema21 else "SHORT"
    sign = 1.0 if direction == "LONG" else -1.0
    move_atr = sign * (c - ctx.last_close) / ctx.atr
    if move_atr < cfg.min_move_atr:
        return _reject("move_too_small", move_atr=round(move_atr, 3))
    if move_atr > cfg.max_move_atr:
        return _reject("already_extended", move_atr=round(move_atr, 3))
    vol_ratio = v / ctx.avg_volume
    need = max(cfg.min_volume_floor, cfg.min_volume_pace * frac)
    if vol_ratio < need:
        return _reject("low_volume_pace", vol_ratio=round(vol_ratio, 3), need=round(need, 3))
    pace = vol_ratio / max(frac, 1e-9)
    ref = ctx.ema21 if ctx.ema21 is not None else ctx.last_close
    pre_extension = sign * (o - ref) / ctx.atr
    if pre_extension > cfg.max_pre_extension_atr:
        return _reject("pre_extended", pre_extension_atr=round(pre_extension, 3))
    structure_break = c > ctx.resistance if direction == "LONG" else c < ctx.support
    impulse = pace >= cfg.impulse_pace and move_atr >= cfg.impulse_move_atr
    # Separate the ignition pathways instead of allowing a lone structural poke
    # or an old squeeze flag to qualify on weak participation. A single volume
    # anomaly is not enough: the move, body/close quality and pace gates above
    # must also pass, and the pathway-specific pace threshold applies here.
    squeeze_ignition = (ctx.compression_age is not None and
                         ctx.compression_age <= cfg.max_squeeze_age_candles and
                         pace >= cfg.min_squeeze_volume_pace)
    if not (squeeze_ignition or impulse):
        return _reject("no_ignition_evidence")

    speed = move_atr / max(elapsed_ms / 60000.0, 1e-9)
    result = {
        "state": "LIVE_EXPLOSION", "strategy": "EXPLOSION", "direction": direction,
        "trigger_price": c, "candle_time_ms": t,
        "candle_close_time_ms": t + ctx.interval_ms,
        "elapsed_seconds": round(elapsed_ms / 1000, 1), "elapsed_fraction": round(frac, 3),
        "atr": ctx.atr, "atr_pct": round(ctx.atr_pct, 4),
        "move_atr": round(move_atr, 3), "speed_atr_per_min": round(speed, 3),
        "volume_ratio": round(vol_ratio, 3), "volume_pace": round(pace, 3),
        "pre_extension_atr": round(pre_extension, 3),
        "structure_break": bool(structure_break), "compression_recent": bool(ctx.compression_recent),
        "squeeze_ignition": bool(squeeze_ignition),
        "impulse": bool(impulse), "high": h, "low": l,
        # Stop loss is ATR-based in the shared risk calculator, not candle-shape based.
        "swing_low": None, "swing_high": None,
        "ema9": ctx.ema9, "ema21": ctx.ema21, "ema50": ctx.ema50,
    }
    result["score_result"] = score_explosion(result, ctx, cfg)
    return result


def score_explosion(res: Dict[str, Any], ctx: LiveContext, cfg: LiveConfig = LiveConfig()) -> ScoreResult:
    d = res["direction"]
    speed, pace, move = res["speed_atr_per_min"], res["volume_pace"], res["move_atr"]
    comps = [
        ("speed", 2.0 if speed >= 1.0 else 1.5 if speed >= 0.6 else 1.0 if speed >= 0.35 else 0.5, 2.0,
         speed, "ATR moved per minute since candle open"),
        ("volume_pace", 2.0 if pace >= 3.0 else 1.5 if pace >= 2.2 else 1.0, 2.0,
         pace, "time-normalised volume vs average candle"),
        ("compression_release", 1.0 if res["compression_recent"] else 0.0, 1.0,
         res["compression_recent"], "BB inside KC within last 6 candles"),
        ("entry_cleanliness", 1.0 if move <= 1.5 else 0.6 if move <= 1.8 else 0.3, 1.0,
         move, "smaller move at trigger = earlier entry"),
        ("trend_alignment", _trend_points(ctx, d, res["trigger_price"]), 1.0,
         None, "EMA9/21 and EMA50 side"),
    ]
    return _build_score(comps, d)


def evaluate_orb(ctx: LiveContext, candle: Dict[str, Any], now_ms: int, tracker: "LiveTracker",
                 cfg: LiveConfig = LiveConfig()) -> List[Dict[str, Any]]:
    """Trigger on the first confirmed break of an opening range, mid-candle."""
    out: List[Dict[str, Any]] = []
    parsed = _candle(candle)
    if parsed is None or not ctx.orb:
        return out
    o, h, l, c, v, t = parsed
    if t != ctx.forming_open_ms:
        return out
    elapsed_ms = now_ms - t
    frac = elapsed_ms / ctx.interval_ms
    for session, info in ctx.orb.items():
        rh, rl = info["range_high"], info["range_low"]
        if not (info["session_start_utc_ms"] <= now_ms < info["session_end_utc_ms"]):
            continue
        buffer = cfg.orb_buffer_atr * ctx.atr
        for direction, beyond, edge in (("LONG", c > rh + buffer, rh), ("SHORT", c < rl - buffer, rl)):
            key = (ctx.pair, session, info["session_date"], direction)
            first = tracker.cross_since(key, now_ms) if beyond else tracker.clear_cross(key)
            if not beyond or first is None:
                continue
            if key in tracker.orb_fired:
                continue
            prev_beyond = ctx.last_close > rh if direction == "LONG" else ctx.last_close < rl
            if prev_beyond:      # continuation of an older break: not an early entry
                continue
            if now_ms - first < cfg.orb_hold_seconds * 1000 or elapsed_ms < cfg.min_elapsed_seconds * 1000:
                continue
            sign = 1.0 if direction == "LONG" else -1.0
            ext = sign * (c - edge) / ctx.atr
            if ext > cfg.orb_max_extension_atr:
                continue
            vol_ratio = v / ctx.avg_volume
            need = max(cfg.orb_min_volume_floor, cfg.orb_min_volume_pace * max(frac, 0.0))
            if vol_ratio < need:
                continue
            res = {
                "state": "LIVE_ORB", "strategy": "ORB_BREAKOUT", "direction": direction,
                "trigger_price": c, "candle_time_ms": t, "candle_close_time_ms": t + ctx.interval_ms,
                "elapsed_seconds": round(elapsed_ms / 1000, 1), "elapsed_fraction": round(frac, 3),
                "atr": ctx.atr, "atr_pct": round(ctx.atr_pct, 4),
                "session": session, "session_date": info["session_date"],
                "session_timezone": info["session_timezone"],
                "session_open_local": info["session_open_local"],
                "range_high": rh, "range_low": rl, "range_width_atr": info["range_width_atr"],
                "breakout_level": edge, "breakout_extension_atr": round(ext, 3),
                "move_atr": round(sign * (c - o) / ctx.atr, 3),
                "volume_ratio": round(vol_ratio, 3),
                "volume_pace": round(vol_ratio / max(frac, 1e-9), 3),
                        "held_seconds": round((now_ms - first) / 1000, 1),
                "high": h, "low": l,
                # Same invalidation as the closed-candle ORB: opposite edge of the range.
                "swing_low": rl if direction == "LONG" else None,
                "swing_high": rh if direction == "SHORT" else None,
                "ema9": ctx.ema9, "ema21": ctx.ema21, "ema50": ctx.ema50,
            }
            res["score_result"] = score_orb(res, ctx, cfg)
            out.append(res)
    return out


def score_orb(res: Dict[str, Any], ctx: LiveContext, cfg: LiveConfig = LiveConfig()) -> ScoreResult:
    d, ext, pace, width = res["direction"], res["breakout_extension_atr"], res["volume_pace"], res["range_width_atr"]
    move = res["move_atr"]
    comps = [
        ("break_extension", 2.0 if ext <= 0.30 else 1.5 if ext <= 0.45 else 1.0, 2.0,
         ext, "distance beyond range edge (smaller = earlier)"),
        ("volume_pace", 2.0 if pace >= 2.5 else 1.5 if pace >= 1.8 else 1.0, 2.0,
         pace, "time-normalised volume vs average candle"),
        ("range_quality", 1.0 if 0.30 <= width <= 1.60 else 0.5, 1.0,
         width, "opening-range width in ATR"),
        ("impulse", 1.5 if move >= 1.0 else 1.0 if move >= 0.6 else 0.5 if move >= 0.3 else 0.0, 1.5,
         move, "ATR moved since candle open"),
        ("trend_alignment", _trend_points(ctx, d, res["trigger_price"]), 1.0,
         None, "EMA9/21 and EMA50 side"),
        ("session", 0.7 if res["session"] == "asia" else 1.0, 1.0,
         res["session"], "London/New York liquidity"),
    ]
    return _build_score(comps, d)


def _build_score(comps: List[Tuple[str, float, float, Any, str]], direction: str) -> ScoreResult:
    components = [
        ScoreComponent(name=n, points=round(p, 2), max_points=m, value=val,
                       direction="bullish" if direction == "LONG" else "bearish", explanation=expl)
        for n, p, m, val, expl in comps
    ]
    return ScoreResult(total=round(sum(x.points for x in components), 2),
                       components=components, direction=direction)


class LiveTracker:
    """Small amount of state: contexts, ORB hold timers, one-shot registries."""

    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self._ctx: Dict[str, LiveContext] = {}
        self._ctx_failed: Dict[str, Tuple[int, int]] = {}
        self._cross: Dict[Tuple[str, str, str, str], int] = {}
        self.orb_fired: set = set()
        self._attempted: set = set()

    # -- ORB hold timers ----------------------------------------------------
    def cross_since(self, key: Tuple[str, str, str, str], now_ms: int) -> int:
        return self._cross.setdefault(key, now_ms)

    def clear_cross(self, key: Tuple[str, str, str, str]) -> None:
        self._cross.pop(key, None)
        return None

    # -- one attempt per pair/direction/candle ---------------------------------
    def first_attempt(self, pair: str, direction: str, candle_ms: int) -> bool:
        key = (pair, direction, int(candle_ms))
        if key in self._attempted:
            return False
        self._attempted.add(key)
        if len(self._attempted) > 5000:
            newest = sorted(self._attempted, key=lambda k: k[2])[-1000:]
            self._attempted = set(newest)
        return True

    # -- context ---------------------------------------------------------------
    def context(self, pair: str, forming_open_ms: int, interval_ms: int, now_ms: int,
                arrays_fn: Callable[[], Optional[Dict[str, Any]]]) -> Optional[LiveContext]:
        ctx = self._ctx.get(pair)
        if ctx is not None and ctx.forming_open_ms == forming_open_ms:
            return ctx
        failed = self._ctx_failed.get(pair)
        if failed and failed[0] == forming_open_ms and now_ms - failed[1] < 20_000:
            return None
        arrays = arrays_fn()
        ctx = build_context(pair, arrays, forming_open_ms, interval_ms, now_ms,
                            self.cfg.sessions, self.cfg.orb_active_minutes) if arrays else None
        if ctx is None:
            self._ctx_failed[pair] = (forming_open_ms, now_ms)
            return None
        self._ctx_failed.pop(pair, None)
        self._ctx[pair] = ctx
        return ctx

    def evaluate(self, pair: str, candle: Dict[str, Any], interval_ms: int, now_ms: int,
                 arrays_fn: Callable[[], Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """All candidates that fire at this instant, best score first."""
        parsed = _candle(candle)
        if parsed is None:
            return []
        t = parsed[5]
        # Only the bucket that is actually forming now: ignore late/replayed events.
        if not (t <= now_ms < t + interval_ms):
            return []
        ctx = self.context(pair, t, interval_ms, now_ms, arrays_fn)
        if ctx is None:
            return []
        found = evaluate_orb(ctx, candle, now_ms, self, self.cfg)
        explosion = evaluate_explosion(ctx, candle, now_ms, self.cfg)
        if explosion.get("state") == "LIVE_EXPLOSION":
            found.append(explosion)
        found.sort(key=lambda r: r["score_result"].total, reverse=True)
        return found
