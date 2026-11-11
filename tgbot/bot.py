"""Telegram bot integration — signal delivery + commands."""
from typing import Optional, List, Any, Dict
import re
import time
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError

from config import settings
from utils.logger import logger
from utils.time_utils import utc_iso


def _esc(text: Any) -> str:
    """Escape Telegram MarkdownV2 special chars. Safe for None."""
    if text is None:
        return "n/a"
    s = str(text)
    # Escape characters that break legacy Markdown
    for ch in ("_", "*", "`", "["):
        s = s.replace(ch, "\\" + ch)
    return s


def _num(val: Any, decimals: int = 4) -> str:
    """Format number safely for Telegram."""
    if val is None:
        return "n/a"
    try:
        f = float(val)
        if abs(f) >= 1000:
            return f"{f:,.2f}"
        if abs(f) >= 1:
            return f"{f:.4f}".rstrip("0").rstrip(".")
        return f"{f:.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(val)


class TelegramBot:
    def __init__(self):
        self.token = settings.telegram_bot_token
        self.app: Optional[Application] = None
        self.bot: Optional[Bot] = None
        self.admin_ids: List[int] = settings.admin_ids
        self.polling_active = False

    async def start(self) -> None:
        if not self.token:
            logger.warning("TELEGRAM_BOT_TOKEN not set — Telegram disabled")
            return
        try:
            # Bound outbound Telegram HTTP latency so one slow recipient cannot
            # hold a signal scan indefinitely. These are transport timeouts, not retries.
            self.app = (
                Application.builder().token(self.token)
                .connect_timeout(5.0)
                .read_timeout(10.0)
                .write_timeout(10.0)
                .pool_timeout(5.0)
                .build()
            )
            self.bot = self.app.bot
            await self.app.initialize()
            await self.app.start()

            # Push-only Telegram mode: this service sends alerts but never calls
            # getUpdates. This prevents polling conflicts when the same token is
            # also used by another bot process or Telegram client.
            self.polling_active = False
            logger.info(
                f"Telegram bot initialized (admins: {self.admin_ids}, "
                "mode: push-only, polling: disabled)"
            )
        except Exception as e:
            logger.exception(f"Telegram start failed: {e}")
            await self.stop()
            self.bot = None

    async def stop(self) -> None:
        # No Telegram updater is started in push-only mode.
        self.polling_active = False
        if self.app and self.app.running:
            try:
                await self.app.stop()
            except Exception as exc:
                logger.warning(f"Telegram application stop note: {exc}")
        if self.app:
            try:
                await self.app.shutdown()
            except Exception as exc:
                logger.warning(f"Telegram application shutdown note: {exc}")
    def _register_handlers(self) -> None:
        assert self.app
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("ping", self.cmd_ping))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("signals", self.cmd_signals))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id if update.effective_user else 0
        await update.message.reply_text(
            f"🚀 Delta Exchange India Signal Bot v3.0\n\n"
            f"Signal-only system. No automated trading.\n"
            f"Your ID: {uid}\n"
            f"Use /help for commands.\n"
            f"Use /ping for connectivity."
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Commands\n"
            "/ping — Silent keepalive (no reply or scan)\n"
            "/status — System health link\n"
            "/signals — Info\n"
            "/help — This message\n\n"
            "Signals are pushed automatically when score ≥ threshold."
        )

    async def cmd_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ping silently; keepalive requests must never send Telegram messages."""
        return None

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Health endpoints: /health · /status"
        )

    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Active signals are pushed live when generated.")

    def format_signal(self, signal: Dict[str, Any]) -> str:
        """Format compact HTML alerts without candle-timing metadata."""
        from html import escape

        def safe(value: Any) -> str:
            return escape(str(value if value is not None else "n/a"), quote=False)

        risk = signal.get("risk") or {}
        ind = signal.get("indicators") or {}
        direction = str(signal.get("direction", "")).upper()
        is_long = direction == "LONG"
        dot = "🟢" if is_long else "🔴"
        side = "LONG" if is_long else "SHORT"
        pair = signal.get("display") or signal.get("pair", "UNKNOWN")
        engine = str(signal.get("engine") or "").upper()
        strategy = signal.get("strategy", "n/a")
        strategy_key = str(strategy or "").upper()
        if engine == "SIGNALS":
            signal_type = "SEQUENTIAL"
        elif strategy_key.startswith("ORB_") or (signal.get("indicators") or {}).get("orb_session"):
            signal_type = "ORB"
        elif engine == "NORMAL":
            signal_type = "NORMAL"
        elif engine == "MOMENTUM":
            signal_type = "MOMENTUM EXPLOSION"
        else:
            signal_type = engine or "N/A"
        score = signal.get("score", 0)
        grade = signal.get("grade", "")
        price = signal.get("price")

        # Keep the alternate sequential-signal engine readable and compact.
        if engine == "SIGNALS":
            step = signal.get("step") or {}
            quality = signal.get("quality") or {}
            rows = [
                f"{dot} <b>{safe(pair)} — {side} SIGNAL</b>",
                f"<b>SIGNAL TYPE:</b> SEQUENTIAL",
                f"<b>ENGINE:</b> SEQUENTIAL",
                f"<b>STRATEGY:</b> {safe(strategy)}",
                f"<b>TIMEFRAME:</b> 5M (PRIMARY)",
                "",
                "<b>ENTRY CONFIRMATIONS</b>",
                f"<b>EMA 9:</b> {safe(_num(ind.get('ema9')))} | <b>EMA 21:</b> {safe(_num(ind.get('ema21')))}",
                f"<b>VWAP:</b> {safe(_num(ind.get('vwap')))}",
                f"<b>RSI:</b> {safe(_num(ind.get('rsi'), 2))} | <b>ADX:</b> {safe(_num(ind.get('adx'), 2))}",
                f"<b>+DI:</b> {safe(_num(ind.get('plus_di'), 2))} | <b>-DI:</b> {safe(_num(ind.get('minus_di'), 2))}",
                "",
                "<b>TRADE LEVELS</b>",
                f"<b>ENTRY:</b> {safe(_num(risk.get('entry')))}",
                f"<b>STOP LOSS:</b> {safe(_num(risk.get('stop_loss')))}",
                f"<b>TP1:</b> {safe(_num(risk.get('tp1')))} | <b>TP2:</b> {safe(_num(risk.get('tp2')))}",
                f"<b>STEP EVENT:</b> {safe(step.get('event', 'n/a'))}",
                f"<b>RVOL:</b> {safe(_num(quality.get('rvol'), 2))}",
                "",
                "<i>Signal only — no automated trading.</i>",
            ]
            return "\n".join(rows)

        rows = [
            f"{dot} <b>{safe(pair)} — {side} SIGNAL</b>",
            f"<b>SIGNAL TYPE:</b> {safe(signal_type)}",
            f"<b>ENGINE:</b> {safe(engine or 'n/a')}",
            f"<b>STRATEGY:</b> {safe(strategy)}",
            *(["<b>ENTRY TYPE:</b> LIVE INTRABAR — early trigger, candle still forming"] if signal.get("entry_type") == "LIVE_INTRABAR" else []),
            *([f"<b>ORB SESSION:</b> {safe(ind.get('orb_session'))} | <b>LOCAL OPEN:</b> {safe(ind.get('orb_session_open_local'))}"] if ind.get("orb_session") else []),
            f"<b>SCORE:</b> {safe(score)}/10 — <b>{safe(grade)}</b>",
            "",
        ]

        if risk:
            ez = risk.get("entry_zone") or [risk.get("entry"), risk.get("entry")]
            rr = risk.get("rr") or [0, 0, 0]
            rows += [
                "<b>TRADE LEVELS</b>",
                f"<b>ENTRY:</b> {safe(_num(risk.get('entry')))}",
                f"<b>LIVE PRICE CHECK:</b> {safe(_num(signal.get('live_price')))}",
                f"<b>ENTRY ZONE:</b> {safe(_num(ez[0]))} – {safe(_num(ez[1]))}",
                f"<b>STOP LOSS:</b> {safe(_num(risk.get('stop_loss')))}",
                f"<b>TP1:</b> {safe(_num(risk.get('tp1')))}",
                f"<b>TP2:</b> {safe(_num(risk.get('tp2')))}",
                f"<b>TP3:</b> {safe(_num(risk.get('tp3')))}",
                f"<b>RISK:</b> {safe(_num(risk.get('risk_pct'), 2))}%",
                f"<b>R:R:</b> 1:{safe(_num(rr[0], 2))} / 1:{safe(_num(rr[1], 2))} / 1:{safe(_num(rr[2], 2))}",
                "",
            ]
        else:
            rows += [f"<b>PRICE:</b> {safe(_num(price))}", ""]

        rows += [
            "<b>MARKET INDICATORS</b>",
            f"<b>TIMEFRAME:</b> 5M (PRIMARY)",
            f"<b>EMA 9/21:</b> {safe(_num(ind.get('ema9')))} / {safe(_num(ind.get('ema21')))}",
            f"<b>RSI:</b> {safe(_num(ind.get('rsi'), 2))} | <b>ADX:</b> {safe(_num(ind.get('adx'), 2))}",
            f"<b>ATR%:</b> {safe(_num(ind.get('atr_pct'), 3))} | <b>RVOL:</b> {safe(_num(ind.get('rvol'), 2))}",
            f"<b>MACD HIST:</b> {safe(_num(ind.get('macd_hist')))}",
            "",
            "<i>Signal only — no automated trading.</i>",
        ]
        return "\n".join(rows)

    async def send_signal_to_chat(self, signal: Dict[str, Any], chat_id: int) -> bool:
        """Deliver one outbox item to exactly one Telegram recipient."""
        if not self.bot:
            raise RuntimeError("Telegram bot is not initialized")
        await self.bot.send_message(
            chat_id=int(chat_id),
            text=self.format_signal(signal),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        return True

    async def send_signal(self, signal: Dict[str, Any]) -> bool:
        """Push a signal to configured admins; return True if at least one delivery succeeds."""
        if not self.bot:
            logger.warning("Telegram bot not ready — signal not sent")
            return False
        if not self.admin_ids:
            logger.warning("TELEGRAM_ADMIN_IDS empty — no recipients")
            return False
        text = self.format_signal(signal)
        success_count = 0
        for chat_id in self.admin_ids:
            recipient_started = time.perf_counter()
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                success_count += 1
                logger.info(
                    f"TELEGRAM_RECIPIENT_DELIVERED chat_id={chat_id} "
                    f"duration_ms={( __import__('time').perf_counter() - recipient_started) * 1000.0:.1f}"
                )
                logger.info(
                    f"Telegram signal sent to {chat_id}: "
                    f"{signal.get('pair')} {signal.get('direction')}"
                )
            except TelegramError as e:
                logger.error(
                    f"TELEGRAM_RECIPIENT_FAILED chat_id={chat_id} "
                    f"duration_ms={( __import__('time').perf_counter() - recipient_started) * 1000.0:.1f} error={e}"
                )
            except Exception as e:
                logger.error(
                    f"TELEGRAM_RECIPIENT_UNEXPECTED_ERROR chat_id={chat_id} "
                    f"duration_ms={( __import__('time').perf_counter() - recipient_started) * 1000.0:.1f} error={e}"
                )
        return success_count > 0
