import pandas as pd

from .config import FEE_RATE, HOLD_PERIOD_CANDLES, STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRADE_SIZE


def simulate_trade_outcome(frame: pd.DataFrame, entry_index: int, side: str) -> float:
    entry_price = float(frame["close"].iloc[entry_index])
    fee = TRADE_SIZE * FEE_RATE

    if side == "LONG":
        tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
        sl_price = entry_price * (1 - STOP_LOSS_PCT)
    else:
        tp_price = entry_price * (1 - TAKE_PROFIT_PCT)
        sl_price = entry_price * (1 + STOP_LOSS_PCT)

    for step in range(1, HOLD_PERIOD_CANDLES + 1):
        if entry_index + step >= len(frame):
            break

        high_price = float(frame["high"].iloc[entry_index + step])
        low_price = float(frame["low"].iloc[entry_index + step])

        if side == "LONG":
            if high_price >= tp_price and low_price > sl_price:
                return TRADE_SIZE * TAKE_PROFIT_PCT - fee
            if low_price <= sl_price and high_price < tp_price:
                return -TRADE_SIZE * STOP_LOSS_PCT - fee
            if high_price >= tp_price and low_price <= sl_price:
                return -TRADE_SIZE * STOP_LOSS_PCT - fee
        else:
            if low_price <= tp_price and high_price < sl_price:
                return TRADE_SIZE * TAKE_PROFIT_PCT - fee
            if high_price >= sl_price and low_price > tp_price:
                return -TRADE_SIZE * STOP_LOSS_PCT - fee
            if low_price <= tp_price and high_price >= sl_price:
                return -TRADE_SIZE * STOP_LOSS_PCT - fee

    exit_price = float(frame["close"].iloc[min(entry_index + HOLD_PERIOD_CANDLES, len(frame) - 1)])
    if side == "LONG":
        change = (exit_price - entry_price) / entry_price
    else:
        change = (entry_price - exit_price) / entry_price
    return TRADE_SIZE * change - fee


def simulate_signal_strategy(frame: pd.DataFrame, probabilities, classes, signal_threshold: float, class_gap: float) -> dict:
    balance = 0.0
    wins = 0
    losses = 0
    trades = 0
    index = 0

    while index < len(frame) - HOLD_PERIOD_CANDLES:
        prob_dict = dict(zip(classes, probabilities[index]))
        long_prob = float(prob_dict.get(1, 0.0))
        short_prob = float(prob_dict.get(-1, 0.0))
        no_trade_prob = float(prob_dict.get(0, 0.0))

        ranked = sorted(
            [("LONG", long_prob), ("SHORT", short_prob), ("NONE", no_trade_prob)],
            key=lambda item: item[1],
            reverse=True,
        )
        side, best_prob = ranked[0]
        second_best_prob = ranked[1][1]

        if side == "NONE" or best_prob < signal_threshold or (best_prob - second_best_prob) < class_gap:
            index += 1
            continue

        profit = simulate_trade_outcome(frame, index, side)
        balance += profit
        trades += 1
        if profit > 0:
            wins += 1
        else:
            losses += 1
        index += HOLD_PERIOD_CANDLES

    win_rate = wins / trades if trades else 0.0
    coverage = trades / max(len(frame), 1)
    avg_profit = balance / trades if trades else 0.0
    return {
        "net_profit": round(balance, 6),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "coverage": coverage,
        "avg_profit": avg_profit,
    }
