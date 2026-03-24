from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class SymbolFilters:
    quantity_precision: int | None
    price_precision: int | None
    step_size: str
    min_qty: str
    tick_size: str
    min_notional: str


@dataclass
class PositionState:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    pnl: float


class ExchangeProvider(ABC):
    provider_name: str
    mode: str
    endpoint: str

    @abstractmethod
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_symbol_filters(self, symbol: str) -> SymbolFilters | None:
        raise NotImplementedError

    @abstractmethod
    def ensure_leverage(self, symbol: str, leverage: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_available_balance(self, asset: str = "USDT") -> float:
        raise NotImplementedError

    @abstractmethod
    def get_open_position(self) -> PositionState | None:
        raise NotImplementedError

    @abstractmethod
    def cancel_all_open_orders(self, symbol: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def place_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def set_protective_orders(self, symbol: str, side: str, take_profit_price: float, stop_loss_price: float) -> bool:
        raise NotImplementedError
