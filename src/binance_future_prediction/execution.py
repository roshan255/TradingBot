from decimal import Decimal, ROUND_DOWN, ROUND_UP
import logging

from .config import STOP_LOSS_PCT, TAKE_PROFIT_PCT
from .exchanges import ExchangeProvider, PositionState, SymbolFilters
from .settings import get_trading_settings


LOGGER = logging.getLogger(__name__)


def _configured_leverage() -> int:
    return int(get_trading_settings().get("default_leverage", 10))


def _configured_fixed_margin_usdt() -> float:
    return float(get_trading_settings().get("fixed_margin_usdt", 10.0))


def _use_full_balance() -> bool:
    return bool(get_trading_settings().get("use_full_account_balance", False))


def _balance_usage_fraction() -> float:
    return float(get_trading_settings().get("balance_usage_fraction", 0.98))


def ensure_symbol_leverage(client: ExchangeProvider, symbol: str) -> None:
    leverage = _configured_leverage()
    if client.ensure_leverage(symbol, leverage):
        LOGGER.info("Set leverage for %s on %s to %sx", symbol, client.provider_name, leverage)


def round_step(value: float, step_size: str, rounding=ROUND_DOWN) -> float:
    step = Decimal(step_size)
    rounded = Decimal(str(value)).quantize(step, rounding=rounding)
    return float(rounded)


def round_price(price: float, tick_size: str) -> float:
    tick = Decimal(tick_size)
    rounded = Decimal(str(price)).quantize(tick, rounding=ROUND_DOWN)
    return float(rounded)


def get_available_usdt_balance(client: ExchangeProvider) -> float:
    return client.get_available_balance("USDT")


def _trade_margin_usdt(client: ExchangeProvider) -> float:
    if _use_full_balance():
        available_balance = get_available_usdt_balance(client)
        return available_balance * _balance_usage_fraction()
    return _configured_fixed_margin_usdt()


def calculate_order_quantity(client: ExchangeProvider, symbol: str, price: float) -> float | None:
    filters = client.get_symbol_filters(symbol)
    if not filters:
        LOGGER.error("Unable to load symbol filters for %s on %s", symbol, client.provider_name)
        return None

    margin_usdt = _trade_margin_usdt(client)
    if margin_usdt <= 0:
        LOGGER.error("No usable USDT balance available for trading")
        return None

    leverage = _configured_leverage()
    notional_usdt = margin_usdt * leverage
    raw_quantity = notional_usdt / price
    quantity = round_step(raw_quantity, filters.step_size, rounding=ROUND_DOWN)
    min_qty = float(filters.min_qty)
    min_notional = float(filters.min_notional)

    if quantity < min_qty:
        quantity = round_step(min_qty, filters.step_size, rounding=ROUND_UP)

    if quantity * price < min_notional:
        quantity = round_step((min_notional / price), filters.step_size, rounding=ROUND_UP)

    LOGGER.info(
        "Sizing trade for %s on %s | margin=%.2f USDT leverage=%sx notional=%.2f USDT qty=%.8f full_balance_mode=%s balance_fraction=%.2f",
        symbol,
        client.provider_name,
        margin_usdt,
        leverage,
        quantity * price,
        quantity,
        _use_full_balance(),
        _balance_usage_fraction(),
    )
    return quantity if quantity > 0 else None


def get_open_position(client: ExchangeProvider) -> PositionState | None:
    return client.get_open_position()


def cancel_symbol_orders(client: ExchangeProvider, symbol: str) -> None:
    client.cancel_all_open_orders(symbol)


def place_entry_with_exits(client: ExchangeProvider, symbol: str, side: str, reference_price: float) -> dict | None:
    ensure_symbol_leverage(client, symbol)

    quantity = calculate_order_quantity(client, symbol, reference_price)
    if quantity is None:
        LOGGER.error("Unable to calculate quantity for %s", symbol)
        return None

    filters = client.get_symbol_filters(symbol)
    if not filters:
        return None

    entry_order = client.place_market_order(symbol, side, quantity, reduce_only=False)
    if not entry_order:
        return None

    executed_price = float(entry_order.get("avgPrice") or entry_order.get("price") or 0.0)
    if executed_price <= 0:
        executed_price = reference_price

    if side == "BUY":
        take_profit_price = executed_price * (1 + TAKE_PROFIT_PCT)
        stop_loss_price = executed_price * (1 - STOP_LOSS_PCT)
    else:
        take_profit_price = executed_price * (1 - TAKE_PROFIT_PCT)
        stop_loss_price = executed_price * (1 + STOP_LOSS_PCT)

    take_profit_price = round_price(take_profit_price, filters.tick_size)
    stop_loss_price = round_price(stop_loss_price, filters.tick_size)

    if not client.set_protective_orders(symbol, side, take_profit_price, stop_loss_price):
        LOGGER.error("Protective orders failed for %s on %s, closing position immediately", symbol, client.provider_name)
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
        "Opened %s %s on %s qty=%.8f entry=%.6f tp=%.6f sl=%.6f leverage=%sx",
        symbol,
        side,
        client.provider_name,
        quantity,
        executed_price,
        take_profit_price,
        stop_loss_price,
        _configured_leverage(),
    )
    return entry_order


def close_position(client: ExchangeProvider, position: PositionState, reason: str) -> dict | None:
    cancel_symbol_orders(client, position.symbol)

    close_side = "SELL" if position.side == "BUY" else "BUY"
    order = client.place_market_order(position.symbol, close_side, position.quantity, reduce_only=True)

    if order:
        LOGGER.info(
            "Closed %s %s on %s qty=%.8f reason=%s pnl=%.6f mark=%.6f",
            position.symbol,
            position.side,
            client.provider_name,
            position.quantity,
            reason,
            position.pnl,
            position.mark_price,
        )

    return order
