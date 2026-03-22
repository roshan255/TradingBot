import json

import joblib
import pandas as pd

from .config import INITIAL_BALANCE, TRAIN_TEST_RATIO, TRAINING_WINDOW_CANDLES
from .paths import get_symbol_file
from .trade_simulation import simulate_signal_strategy
from .universe import FEATURE_COLUMNS


def backtest_symbol(symbol: str) -> dict:
    model = joblib.load(get_symbol_file(symbol, "model.pkl"))
    with get_symbol_file(symbol, "model_meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    signal_rules = meta.get("signal_rules", {})
    signal_threshold = float(signal_rules.get("signal_threshold", 0.34))
    class_gap = float(signal_rules.get("class_gap", 0.00))

    full_df = pd.read_csv(get_symbol_file(symbol, "features.csv"))
    window_rows = int(meta.get("training_window_rows", meta.get("rows", TRAINING_WINDOW_CANDLES)))
    df = full_df.tail(window_rows).reset_index(drop=True)
    split_index = int(len(df) * TRAIN_TEST_RATIO)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    X_test = test_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    probabilities = model.predict_proba(X_test)
    result = simulate_signal_strategy(test_df, probabilities, model.classes_, signal_threshold, class_gap)

    return {
        "symbol": symbol,
        "final_balance": round(INITIAL_BALANCE + result["net_profit"], 2),
        "trades": result["trades"],
        "wins": result["wins"],
        "losses": result["losses"],
        "win_rate": round(result["win_rate"], 2) if result["trades"] else 0,
        "test_rows": len(test_df),
        "signal_threshold": signal_threshold,
        "class_gap": class_gap,
        "avg_profit": round(result["avg_profit"], 4),
    }
