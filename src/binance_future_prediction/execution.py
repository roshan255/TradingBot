from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import logging

from binance.client import Client
from binance.exceptions import BinanceAPIException

from .config import STOP_LOSS_PCT, TAKE_PROFIT_PCT
from .settings import get_trading_settings


LOGGER = logging.getLogger(__name__)
_SYMBOL_FILTER_CACHE = {}
_LEVERAGE_CACHE = {}


@dataclass
class PositionState:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    pnl: float


def safe_api_call(action: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except BinanceAPIException as exc:
        LOGGER.error("%s failed with Binance API error: %s", action, exc)
    except Exception as exc:
        LOGGER.exception("%s failed: %s", action, exc)
    return None


def _load_symbol_filters(client: Client) -> None:
    if _SYMBOL_FILTER_CACHE:
        return

    exchange_info = safe_api_call("futures_exchange_info", client.futures_exchange_info)
    if not exchange_info:
        return

    for item in exchange_info["symbols"]:
        filters = {flt["filterType"]: flt for flt in item["filters"]}
        _SYMBOL_FILTER_CACHE[item["symbol"]] = {
            "quantity_precision": item["quantityPrecision"],
            "price_precision": item["pricePrecision"],
            "step_size": filters["LOT_SIZE"]["stepSize"],
            "min_qty": filters["LOT_SIZE"]["minQty"],
            "tick_size": filters["PRICE_FILTER"]["tickSize"],
            "min_notional": filters.get("MIN_NOTIONAL", {}).get("notional", "5"),
        }


def get_symbol_filters(client: Client, symbol: str) -> dict | None:
    _load_symbol_filters(client)
    filters = _SYMBOL_FILTER_CACHE.get(symbol)
    if filters is None:
        LOGGER.error("Symbol metadata not found for %s", symbol)
    return filters


def _configured_leverage() -> int:
    return int(get_trading_settings().get("default_leverage", 10))


def _configured_fixed_margin_usdt() -> float:
    return float(get_trading_settings().get("fixed_margin_usdt", 10.0))


def _use_full_balance() -> bool:
    return bool(get_trading_settings().get("use_full_account_balance", False))


def _balance_usage_fraction() -> float:
    return float(get_trading_settings().get("balance_usage_fraction", 0.98))


def ensure_symbol_leverage(client: Client, symbol: str) -> None:
    leverage = _configured_leverage()
    if _LEVERAGE_CACHE.get(symbol) == leverage:
        return

    response = safe_api_call(
        "futures_change_leverage",
        client.futures_change_leverage,
        symbol=symbol,
        leverage=leverage,
    )
    if response:
        _LEVERAGE_CACHE[symbol] = leverage
        LOGGER.info("Set leverage for %s to %sx", symbol, leverage)


def round_step(value: float, step_size: str, rounding=ROUND_DOWN) -> float:
    step = Decimal(step_size)
    rounded = Decimal(str(value)).quantize(step, rounding=rounding)
    return float(rounded)


def round_price(price: float, tick_size: str) -> float:
    tick = Decimal(tick_size)
    rounded = Decimal(str(price)).quantize(tick, rounding=ROUND_DOWN)
    return float(rounded)


def get_available_usdt_balance(client: Client) -> float:
    balances = safe_api_call("futures_account_balance", client.futures_account_balance)
    if not balances:
        return 0.0

    for balance in balances:
        if balance.get("asset") == "USDT":
            return float(balance.get("availableBalance", 0.0))
    return 0.0


def _trade_margin_usdt(client: Client) -> float:
    if _use_full_balance():
        available_balance = get_available_usdt_balance(client)
        return available_balance * _balance_usage_fraction()
    return _configured_fixed_margin_usdt()


def calculate_order_quantity(client: Client, symbol: str, price: float) -> float | None:
    filters = get_symbol_filters(client, symbol)
    if not filters:
        return None

    margin_usdt = _trade_margin_usdt(client)
    if margin_usdt <= 0:
        LOGGER.error("No usable USDT balance available for trading")
        return None

    leverage = _configured_leverage()
    notional_usdt = margin_usdt * leverage
    raw_quantity = notional_usdt / price
    quantity = round_step(raw_quantity, filters["step_size"], rounding=ROUND_DOWN)
    min_qty = float(filters["min_qty"])
    min_notional = float(filters["min_notional"])

    if quantity < min_qty:
        quantity = round_step(min_qty, filters["step_size"], rounding=ROUND_UP)

    if quantity * price < min_notional:
        quantity = round_step((min_notional / price), filters["step_size"], rounding=ROUND_UP)

    LOGGER.info(
        "Sizing trade for %s | margin=%.2f USDT leverage=%sx notional=%.2f USDT qty=%.8f full_balance_mode=%s balance_fraction=%.2f",
        symbol,
        margin_usdt,
        leverage,
        quantity * price,
        quantity,
        _use_full_balance(),
        _balance_usage_fraction(),
    )
    return quantity if quantity > 0 else None


def get_open_position(client: Client) -> PositionState | None:
    positions = safe_api_call("futures_position_information", client.futures_position_information)
    if not positions:
        return None

    for position in positions:
        quantity = float(position["positionAmt"])
        if quantity == 0:
            continue

        side = "BUY" if quantity > 0 else "SELL"
        return PositionState(
            symbol=position["symbol"],
            side=side,
            quantity=abs(quantity),
            entry_price=float(position["entryPrice"]),
            mark_price=float(position["markPrice"]),
            pnl=float(position["unRealizedProfit"]),
        )

    return None


def cancel_symbol_orders(client: Client, symbol: str) -> None:
    safe_api_call("futures_cancel_all_open_orders", client.futures_cancel_all_open_orders, symbol=symbol)


def place_entry_with_exits(client: Client, symbol: str, side: str, reference_price: float) -> dict | None:
    ensure_symbol_leverage(client, symbol)

    quantity = calculate_order_quantity(client, symbol, reference_price)
    if quantity is None:
        LOGGER.error("Unable to calculate quantity for %s", symbol)
        return None

    filters = get_symbol_filters(client, symbol)
    if not filters:
        return None

    entry_order = safe_api_call(
        "futures_create_order(entry)",
        client.futures_create_order,
        symbol=symbol,
        side=side,
        type="MARKET",
        quantity=quantity,
    )
    if not entry_order:
        return None

    exit_side = "SELL" if side == "BUY" else "BUY"
    executed_price = float(entry_order.get("avgPrice") or 0.0)
    if executed_price <= 0:
        executed_price = reference_price

    if side == "BUY":
        take_profit_price = executed_price * (1 + TAKE_PROFIT_PCT)
        stop_loss_price = executed_price * (1 - STOP_LOSS_PCT)
    else:
        take_profit_price = executed_price * (1 - TAKE_PROFIT_PCT)
        stop_loss_price = executed_price * (1 + STOP_LOSS_PCT)

    take_profit_price = round_price(take_profit_price, filters["tick_size"])
    stop_loss_price = round_price(stop_loss_price, filters["tick_size"])

    tp_order = safe_api_call(
        "futures_create_order(take_profit)",
        client.futures_create_order,
        symbol=symbol,
        side=exit_side,
        type="TAKE_PROFIT_MARKET",
        stopPrice=take_profit_price,
        closePosition=True,
        workingType="MARK_PRICE",
    )

    sl_order = safe_api_call(
        "futures_create_order(stop_loss)",
        client.futures_create_order,
        symbol=symbol,
        side=exit_side,
        type="STOP_MARKET",
        stopPrice=stop_loss_price,
        closePosition=True,
        workingType="MARK_PRICE",
    )

    if not tp_order or not sl_order:
        LOGGER.error("Protective orders failed for %s, closing position immediately", symbol)
        close_position(
            client,
            PositionState(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=executed_price,
                mark_price=executed_price,
                pnl=0.0,
            ),
            reason="protective_order_failure",
        )
        return None

    LOGGER.info(
        "Opened %s %s qty=%.8f entry=%.6f tp=%.6f sl=%.6f leverage=%sx",
        symbol,
        side,
        quantity,
        executed_price,
        take_profit_price,
        stop_loss_price,
        _configured_leverage(),
    )
    return entry_order


def close_position(client: Client, position: PositionState, reason: str) -> dict | None:
    cancel_symbol_orders(client, position.symbol)

    close_side = "SELL" if position.side == "BUY" else "BUY"
    order = safe_api_call(
        f"close_position({reason})",
        client.futures_create_order,
        symbol=position.symbol,
        side=close_side,
        type="MARKET",
        quantity=position.quantity,
        reduceOnly=True,
    )

    if order:
        LOGGER.info(
            "Closed %s %s qty=%.8f reason=%s pnl=%.6f mark=%.6f",
            position.symbol,
            position.side,
            position.quantity,
            reason,
            position.pnl,
            position.mark_price,
        )

    return order
