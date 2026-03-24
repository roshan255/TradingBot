import logging
from typing import Any

import pandas as pd

from .base import ExchangeProvider, PositionState, SymbolFilters


LOGGER = logging.getLogger(__name__)


class BinanceExchangeProvider(ExchangeProvider):
    def __init__(self, provider_settings: dict, require_credentials: bool = False, public_only: bool = False):
        from binance.client import Client

        self.provider_name = "binance"
        self.provider_settings = provider_settings
        self.public_only = public_only
        self.mode = provider_settings.get("market_data_mode", "production") if public_only else provider_settings.get("mode", "testnet")
        mode_settings = provider_settings.get(self.mode, {})
        api_key = mode_settings.get("api_key") if require_credentials or not public_only else None
        api_secret = mode_settings.get("api_secret") if require_credentials or not public_only else None
        self.endpoint = mode_settings.get("futures_url", "")
        self.client = Client(api_key, api_secret, testnet=(self.mode == "testnet"))
        if self.endpoint:
            self.client.FUTURES_URL = self.endpoint
        self._symbol_filters: dict[str, SymbolFilters] = {}
        self._leverage_cache: dict[str, int] = {}

    def _safe_call(self, action: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            LOGGER.error("%s failed on Binance: %s", action, exc)
        return None

    def _format_klines(self, klines) -> pd.DataFrame:
        df = pd.DataFrame(klines)
        if df.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = df[[0, 1, 2, 3, 4, 5]]
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for column in ["open", "high", "low", "close", "volume"]:
            df[column] = df[column].astype(float)
        return df.sort_values("time").reset_index(drop=True)

    def fetch_klines(self, symbol: str, interval: str, limit: int, start_time_ms: int | None = None, end_time_ms: int | None = None) -> pd.DataFrame:
        payload: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time_ms is not None:
            payload["startTime"] = start_time_ms
        if end_time_ms is not None:
            payload["endTime"] = end_time_ms
        klines = self._safe_call("futures_klines", self.client.futures_klines, **payload)
        if klines is None:
            raise RuntimeError(f"Unable to fetch Binance klines for {symbol}")
        return self._format_klines(klines)

    def get_symbol_filters(self, symbol: str) -> SymbolFilters | None:
        if symbol in self._symbol_filters:
            return self._symbol_filters[symbol]

        exchange_info = self._safe_call("futures_exchange_info", self.client.futures_exchange_info)
        if not exchange_info:
            return None

        for item in exchange_info.get("symbols", []):
            filters = {flt["filterType"]: flt for flt in item.get("filters", [])}
            self._symbol_filters[item["symbol"]] = SymbolFilters(
                quantity_precision=item.get("quantityPrecision"),
                price_precision=item.get("pricePrecision"),
                step_size=filters.get("LOT_SIZE", {}).get("stepSize", "0.001"),
                min_qty=filters.get("LOT_SIZE", {}).get("minQty", "0.001"),
                tick_size=filters.get("PRICE_FILTER", {}).get("tickSize", "0.0001"),
                min_notional=filters.get("MIN_NOTIONAL", {}).get("notional", "5"),
            )
        filters = self._symbol_filters.get(symbol)
        if filters is None:
            LOGGER.error("Binance symbol metadata not found for %s", symbol)
        return filters

    def ensure_leverage(self, symbol: str, leverage: int) -> bool:
        if self._leverage_cache.get(symbol) == leverage:
            return True
        response = self._safe_call(
            "futures_change_leverage",
            self.client.futures_change_leverage,
            symbol=symbol,
            leverage=leverage,
        )
        if response is None:
            return False
        self._leverage_cache[symbol] = leverage
        return True

    def get_available_balance(self, asset: str = "USDT") -> float:
        balances = self._safe_call("futures_account_balance", self.client.futures_account_balance)
        if not balances:
            return 0.0
        for balance in balances:
            if balance.get("asset") == asset:
                return float(balance.get("availableBalance", 0.0))
        return 0.0

    def get_open_position(self) -> PositionState | None:
        positions = self._safe_call("futures_position_information", self.client.futures_position_information)
        if not positions:
            return None
        for position in positions:
            quantity = float(position.get("positionAmt", 0.0))
            if quantity == 0:
                continue
            return PositionState(
                symbol=position["symbol"],
                side="BUY" if quantity > 0 else "SELL",
                quantity=abs(quantity),
                entry_price=float(position.get("entryPrice", 0.0)),
                mark_price=float(position.get("markPrice", 0.0)),
                pnl=float(position.get("unRealizedProfit", 0.0)),
            )
        return None

    def cancel_all_open_orders(self, symbol: str) -> bool:
        response = self._safe_call("futures_cancel_all_open_orders", self.client.futures_cancel_all_open_orders, symbol=symbol)
        return response is not None

    def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> dict | None:
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        }
        if reduce_only:
            payload["reduceOnly"] = True
        return self._safe_call("futures_create_order", self.client.futures_create_order, **payload)

    def set_protective_orders(self, symbol: str, side: str, take_profit_price: float, stop_loss_price: float) -> bool:
        exit_side = "SELL" if side == "BUY" else "BUY"
        tp_order = self._safe_call(
            "futures_create_order(take_profit)",
            self.client.futures_create_order,
            symbol=symbol,
            side=exit_side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=take_profit_price,
            closePosition=True,
            workingType="MARK_PRICE",
        )
        sl_order = self._safe_call(
            "futures_create_order(stop_loss)",
            self.client.futures_create_order,
            symbol=symbol,
            side=exit_side,
            type="STOP_MARKET",
            stopPrice=stop_loss_price,
            closePosition=True,
            workingType="MARK_PRICE",
        )
        return tp_order is not None and sl_order is not None
