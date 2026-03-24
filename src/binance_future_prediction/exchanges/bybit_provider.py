import logging
from decimal import Decimal

import pandas as pd

from .base import ExchangeProvider, PositionState, SymbolFilters


LOGGER = logging.getLogger(__name__)
_INTERVAL_MAP = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
}


class BybitExchangeProvider(ExchangeProvider):
    def __init__(self, provider_settings: dict, require_credentials: bool = False, public_only: bool = False):
        try:
            from pybit.unified_trading import HTTP
        except ImportError as exc:
            raise ImportError("pybit is required for the Bybit provider. Install it with `pip install pybit`.") from exc

        self.provider_name = "bybit"
        self.provider_settings = provider_settings
        self.public_only = public_only
        self.mode = provider_settings.get("market_data_mode", "production") if public_only else provider_settings.get("mode", "testnet")
        mode_settings = provider_settings.get(self.mode, {})
        api_key = mode_settings.get("api_key") if require_credentials or not public_only else None
        api_secret = mode_settings.get("api_secret") if require_credentials or not public_only else None
        self.endpoint = mode_settings.get("futures_url", "")
        self.category = provider_settings.get("category", "linear")
        self.settle_coin = provider_settings.get("settle_coin", "USDT")
        self.account_type = provider_settings.get("account_type", "UNIFIED")
        self.position_idx = int(provider_settings.get("position_idx", 0))
        self.session = HTTP(testnet=(self.mode == "testnet"), api_key=api_key, api_secret=api_secret)
        self._symbol_filters: dict[str, SymbolFilters] = {}
        self._leverage_cache: dict[str, int] = {}

    def _call(self, action: str, func, *args, acceptable_ret_codes: set[int] | None = None, **kwargs):
        acceptable_ret_codes = acceptable_ret_codes or set()
        try:
            response = func(*args, **kwargs)
        except Exception as exc:
            LOGGER.error("%s failed on Bybit: %s", action, exc)
            return None

        ret_code = int(response.get("retCode", 0))
        ret_msg = response.get("retMsg", "")
        if ret_code != 0 and ret_code not in acceptable_ret_codes:
            LOGGER.error("%s failed on Bybit: retCode=%s retMsg=%s", action, ret_code, ret_msg)
            return None
        return response

    def _interval(self, interval: str) -> str:
        return _INTERVAL_MAP.get(interval, interval)

    def _precision_from_step(self, value: str) -> int | None:
        try:
            decimal_value = Decimal(str(value)).normalize()
        except Exception:
            return None
        exponent = decimal_value.as_tuple().exponent
        return abs(exponent) if exponent < 0 else 0

    def _format_klines(self, payload: list[list[str]]) -> pd.DataFrame:
        df = pd.DataFrame(payload)
        if df.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = df.iloc[:, :6]
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms")
        for column in ["open", "high", "low", "close", "volume"]:
            df[column] = df[column].astype(float)
        return df.sort_values("time").reset_index(drop=True)

    def fetch_klines(self, symbol: str, interval: str, limit: int, start_time_ms: int | None = None, end_time_ms: int | None = None) -> pd.DataFrame:
        payload = {
            "category": self.category,
            "symbol": symbol,
            "interval": self._interval(interval),
            "limit": limit,
        }
        if start_time_ms is not None:
            payload["start"] = start_time_ms
        if end_time_ms is not None:
            payload["end"] = end_time_ms

        response = self._call("get_kline", self.session.get_kline, **payload)
        if response is None:
            raise RuntimeError(f"Unable to fetch Bybit klines for {symbol}")
        return self._format_klines(response.get("result", {}).get("list", []))

    def get_symbol_filters(self, symbol: str) -> SymbolFilters | None:
        if symbol in self._symbol_filters:
            return self._symbol_filters[symbol]

        response = self._call("get_instruments_info", self.session.get_instruments_info, category=self.category, symbol=symbol)
        if response is None:
            return None

        instruments = response.get("result", {}).get("list", [])
        if not instruments:
            LOGGER.error("Bybit symbol metadata not found for %s", symbol)
            return None

        item = instruments[0]
        lot_filter = item.get("lotSizeFilter", {})
        price_filter = item.get("priceFilter", {})
        filters = SymbolFilters(
            quantity_precision=self._precision_from_step(lot_filter.get("qtyStep", "1")),
            price_precision=self._precision_from_step(price_filter.get("tickSize", "1")),
            step_size=lot_filter.get("qtyStep", "1"),
            min_qty=lot_filter.get("minOrderQty", lot_filter.get("minTradingQty", "1")),
            tick_size=price_filter.get("tickSize", "0.0001"),
            min_notional=lot_filter.get("minNotionalValue", "5"),
        )
        self._symbol_filters[symbol] = filters
        return filters

    def ensure_leverage(self, symbol: str, leverage: int) -> bool:
        if self._leverage_cache.get(symbol) == leverage:
            return True
        response = self._call(
            "set_leverage",
            self.session.set_leverage,
            category=self.category,
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage),
            acceptable_ret_codes={110043},
        )
        if response is None:
            return False
        self._leverage_cache[symbol] = leverage
        return True

    def get_available_balance(self, asset: str = "USDT") -> float:
        response = self._call(
            "get_wallet_balance",
            self.session.get_wallet_balance,
            accountType=self.account_type,
            coin=asset,
        )
        if response is None:
            return 0.0

        wallet_list = response.get("result", {}).get("list", [])
        if not wallet_list:
            return 0.0

        account = wallet_list[0]
        total_available = account.get("totalAvailableBalance")
        if total_available not in {None, ""}:
            try:
                return float(total_available)
            except Exception:
                pass

        for coin_info in account.get("coin", []):
            if coin_info.get("coin") == asset:
                for key in ["availableToWithdraw", "walletBalance", "equity"]:
                    value = coin_info.get(key)
                    if value not in {None, ""}:
                        return float(value)
        return 0.0

    def get_open_position(self) -> PositionState | None:
        response = self._call(
            "get_positions",
            self.session.get_positions,
            category=self.category,
            settleCoin=self.settle_coin,
        )
        if response is None:
            return None

        for position in response.get("result", {}).get("list", []):
            quantity = float(position.get("size", 0.0) or 0.0)
            if quantity == 0:
                continue
            side = "BUY" if str(position.get("side", "")).lower() == "buy" else "SELL"
            return PositionState(
                symbol=position.get("symbol", ""),
                side=side,
                quantity=abs(quantity),
                entry_price=float(position.get("avgPrice", 0.0) or 0.0),
                mark_price=float(position.get("markPrice", 0.0) or 0.0),
                pnl=float(position.get("unrealisedPnl", 0.0) or 0.0),
            )
        return None

    def cancel_all_open_orders(self, symbol: str) -> bool:
        response = self._call("cancel_all_orders", self.session.cancel_all_orders, category=self.category, symbol=symbol)
        return response is not None

    def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> dict | None:
        return self._call(
            "place_order",
            self.session.place_order,
            category=self.category,
            symbol=symbol,
            side="Buy" if side == "BUY" else "Sell",
            orderType="Market",
            qty=str(quantity),
            reduceOnly=reduce_only,
            positionIdx=self.position_idx,
        )

    def set_protective_orders(self, symbol: str, side: str, take_profit_price: float, stop_loss_price: float) -> bool:
        response = self._call(
            "set_trading_stop",
            self.session.set_trading_stop,
            category=self.category,
            symbol=symbol,
            takeProfit=str(take_profit_price),
            stopLoss=str(stop_loss_price),
            tpTriggerBy="MarkPrice",
            slTriggerBy="MarkPrice",
            tpslMode="Full",
            tpOrderType="Market",
            slOrderType="Market",
            positionIdx=self.position_idx,
        )
        return response is not None
