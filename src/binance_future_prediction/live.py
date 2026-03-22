from dataclasses import dataclass
from datetime import datetime

import joblib

from .config import INTERVAL, MIN_LIVE_VOLATILITY, SIGNAL_THRESHOLD
from .data import get_latest_klines
from .features import create_feature_frame
from .paths import get_symbol_file
from .universe import FEATURE_COLUMNS, SYMBOLS


@dataclass
class Signal:
    symbol: str
    current_time: str
    price: float
    probability: float
    prediction: str


def generate_live_signals() -> list[Signal]:
    results: list[Signal] = []

    for symbol in SYMBOLS:
        model = joblib.load(get_symbol_file(symbol, "model.pkl"))
        latest_df = get_latest_klines(symbol, INTERVAL)
        feature_df = create_feature_frame(latest_df, include_target=False)

        if feature_df.empty:
            print(symbol, "Not enough candles for indicators")
            continue

        latest = feature_df.iloc[-1:]
        volatility = latest["atr"].iat[0] / latest["close"].iat[0]
        if volatility < MIN_LIVE_VOLATILITY:
            continue

        X = latest[FEATURE_COLUMNS]
        probs = model.predict_proba(X)[0]
        prob_dict = dict(zip(model.classes_, probs))

        short_prob = prob_dict.get(-1, 0)
        no_move_prob = prob_dict.get(0, 0)
        long_prob = prob_dict.get(1, 0)

        prediction = "NO TRADE"
        probability = 0.0

        if long_prob > short_prob and long_prob > no_move_prob and long_prob > SIGNAL_THRESHOLD:
            prediction = ">>> LONG SIGNAL <<<"
            probability = long_prob
        elif short_prob > long_prob and short_prob > no_move_prob and short_prob > SIGNAL_THRESHOLD:
            prediction = ">>> SHORT SIGNAL <<<"
            probability = short_prob

        ma50 = latest["ma50"].iat[0]
        ma200 = latest["ma200"].iat[0]

        if prediction == ">>> LONG SIGNAL <<<" and ma50 < ma200:
            continue
        if prediction == ">>> SHORT SIGNAL <<<" and ma50 > ma200:
            continue
        if prediction == "NO TRADE":
            continue

        results.append(
            Signal(
                symbol=symbol,
                current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                price=float(latest["close"].iat[0]),
                probability=float(probability),
                prediction=prediction,
            )
        )

    return sorted(results, key=lambda item: item.probability, reverse=True)

