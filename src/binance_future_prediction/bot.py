from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time

from binance.client import Client

from .config import INTERVAL, MONITOR_INTERVAL_SECONDS, SCAN_INTERVAL_SECONDS
from .data import get_latest_klines
from .execution import PositionState, close_position, get_open_position, place_entry_with_exits
from .prediction import find_best_trade, predict_symbol
from .settings import get_trading_settings
from .universe import SYMBOLS


LOGGER = logging.getLogger(__name__)


@dataclass
class ActiveTrade:
    symbol: str
    side: str
    opened_at: datetime
    last_checked_at: datetime


class AutoTradingBot:
    def __init__(self, client: Client):
        self.client = client
        self.active_trade: ActiveTrade | None = None
        self.last_scan_candle: datetime | None = None

    def sync_position(self) -> PositionState | None:
        position = get_open_position(self.client)
        if position is None:
            self.active_trade = None
            return None

        if self.active_trade is None or self.active_trade.symbol != position.symbol:
            self.active_trade = ActiveTrade(
                symbol=position.symbol,
                side=position.side,
                opened_at=datetime.now(),
                last_checked_at=datetime.now(),
            )
        return position

    def _latest_closed_candle_time(self) -> datetime | None:
        symbol = SYMBOLS[0]
        df = get_latest_klines(symbol, INTERVAL, limit=2, client=self.client)
        if len(df) < 2:
            return None
        return df["time"].iloc[-2].to_pydatetime()

    def run_forever(self) -> None:
        trading_settings = get_trading_settings()
        LOGGER.info(
            "Auto trading bot started | leverage=%sx fixed_margin=%.2f use_full_balance=%s balance_fraction=%.2f",
            trading_settings.get("default_leverage", 10),
            trading_settings.get("fixed_margin_usdt", 10.0),
            trading_settings.get("use_full_account_balance", False),
            trading_settings.get("balance_usage_fraction", 0.98),
        )

        while True:
            try:
                position = self.sync_position()
                if position is None:
                    self.scan_and_trade()
                else:
                    self.monitor_position(position)
            except KeyboardInterrupt:
                LOGGER.info("Bot stopped by user")
                break
            except Exception as exc:
                LOGGER.exception("Bot loop error: %s", exc)
                time.sleep(SCAN_INTERVAL_SECONDS)

    def scan_and_trade(self) -> None:
        latest_closed_candle = self._latest_closed_candle_time()
        if latest_closed_candle is None:
            LOGGER.info("Waiting for enough candle data before scanning")
            time.sleep(SCAN_INTERVAL_SECONDS)
            return

        if self.last_scan_candle == latest_closed_candle:
            time.sleep(SCAN_INTERVAL_SECONDS)
            return

        self.last_scan_candle = latest_closed_candle
        LOGGER.info("Scanning new closed candle at %s", latest_closed_candle.strftime("%Y-%m-%d %H:%M:%S"))

        candidate = find_best_trade(self.client)
        if candidate is None:
            return

        LOGGER.info(
            "Best trade selected symbol=%s side=%s prob=%.3f price=%.6f threshold=%.2f gap=%.2f context=%.2f headline=%s",
            candidate.symbol,
            candidate.side,
            candidate.probability,
            candidate.price,
            candidate.signal_threshold,
            candidate.class_gap_threshold,
            candidate.context_score,
            candidate.context_headline,
        )

        order = place_entry_with_exits(self.client, candidate.symbol, candidate.side, candidate.price)
        if not order:
            time.sleep(SCAN_INTERVAL_SECONDS)
            return

        now = datetime.now()
        self.active_trade = ActiveTrade(
            symbol=candidate.symbol,
            side=candidate.side,
            opened_at=now,
            last_checked_at=now,
        )
        time.sleep(SCAN_INTERVAL_SECONDS)

    def monitor_position(self, position: PositionState) -> None:
        LOGGER.info(
            "Open position symbol=%s side=%s entry=%.6f mark=%.6f pnl=%.6f",
            position.symbol,
            position.side,
            position.entry_price,
            position.mark_price,
            position.pnl,
        )

        if self.active_trade is None:
            self.active_trade = ActiveTrade(
                symbol=position.symbol,
                side=position.side,
                opened_at=datetime.now(),
                last_checked_at=datetime.now(),
            )

        next_check = self.active_trade.last_checked_at + timedelta(seconds=MONITOR_INTERVAL_SECONDS)
        if datetime.now() < next_check:
            time.sleep(SCAN_INTERVAL_SECONDS)
            return

        candidate = predict_symbol(self.client, position.symbol)
        self.active_trade.last_checked_at = datetime.now()

        if candidate is None:
            LOGGER.info("Monitoring check kept %s because no opposite signal appeared", position.symbol)
            time.sleep(SCAN_INTERVAL_SECONDS)
            return

        LOGGER.info(
            "Monitoring prediction symbol=%s side=%s prob=%.3f context=%.2f",
            candidate.symbol,
            candidate.side,
            candidate.probability,
            candidate.context_score,
        )

        if candidate.side != position.side:
            close_position(self.client, position, reason="prediction_flip")
            self.active_trade = None
        else:
            LOGGER.info("Position kept open for %s", position.symbol)

        time.sleep(SCAN_INTERVAL_SECONDS)
