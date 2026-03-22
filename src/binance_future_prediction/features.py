import numpy as np
import pandas as pd
import ta

from .config import (
    HOLD_PERIOD_CANDLES,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TARGET_DOMINANCE_RATIO,
    TARGET_EXTENSION_MULTIPLIER,
    TARGET_MAX_HIT_CANDLES,
    TARGET_MIN_ATR_PCT,
)
from .universe import FEATURE_COLUMNS


def _build_trade_targets(frame: pd.DataFrame) -> pd.Series:
    targets = np.zeros(len(frame), dtype=int)
    closes = frame["close"].to_numpy()
    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    atr_values = frame["atr"].to_numpy() if "atr" in frame.columns else np.full(len(frame), np.nan)

    for index in range(len(frame) - HOLD_PERIOD_CANDLES):
        entry = closes[index]
        atr_pct = float(atr_values[index] / entry) if np.isfinite(atr_values[index]) and entry else 0.0
        if atr_pct < TARGET_MIN_ATR_PCT:
            continue

        long_tp = entry * (1 + TAKE_PROFIT_PCT)
        long_sl = entry * (1 - STOP_LOSS_PCT)
        short_tp = entry * (1 - TAKE_PROFIT_PCT)
        short_sl = entry * (1 + STOP_LOSS_PCT)

        max_up = 0.0
        max_down = 0.0
        long_step = None
        short_step = None

        for step in range(1, HOLD_PERIOD_CANDLES + 1):
            high_price = highs[index + step]
            low_price = lows[index + step]
            max_up = max(max_up, (high_price - entry) / entry)
            max_down = max(max_down, (entry - low_price) / entry)

            if long_step is None:
                hit_long_tp = high_price >= long_tp
                hit_long_sl = low_price <= long_sl
                if hit_long_tp and not hit_long_sl:
                    long_step = step
                elif hit_long_tp and hit_long_sl:
                    long_step = -1

            if short_step is None:
                hit_short_tp = low_price <= short_tp
                hit_short_sl = high_price >= short_sl
                if hit_short_tp and not hit_short_sl:
                    short_step = step
                elif hit_short_tp and hit_short_sl:
                    short_step = -1

            if long_step not in {None, -1} and short_step not in {None, -1}:
                break

        long_quality = (
            long_step not in {None, -1}
            and short_step in {None, -1}
            and long_step <= TARGET_MAX_HIT_CANDLES
            and max_up >= TAKE_PROFIT_PCT * TARGET_EXTENSION_MULTIPLIER
            and max_up > max_down * TARGET_DOMINANCE_RATIO
        )
        short_quality = (
            short_step not in {None, -1}
            and long_step in {None, -1}
            and short_step <= TARGET_MAX_HIT_CANDLES
            and max_down >= TAKE_PROFIT_PCT * TARGET_EXTENSION_MULTIPLIER
            and max_down > max_up * TARGET_DOMINANCE_RATIO
        )

        if long_quality and not short_quality:
            targets[index] = 1
        elif short_quality and not long_quality:
            targets[index] = -1

    targets[len(frame) - HOLD_PERIOD_CANDLES:] = 0
    return pd.Series(targets, index=frame.index)


def create_feature_frame(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    price = frame["close"]

    frame["ma20"] = ta.trend.sma_indicator(price, window=20)
    frame["ma50"] = ta.trend.sma_indicator(price, window=50)
    frame["ma200"] = ta.trend.sma_indicator(price, window=200)
    frame["ema20"] = ta.trend.ema_indicator(price, window=20)
    frame["ema50"] = ta.trend.ema_indicator(price, window=50)

    frame["rsi"] = ta.momentum.rsi(price, window=14)

    macd = ta.trend.MACD(price)
    frame["macd"] = macd.macd()
    frame["macd_signal"] = macd.macd_signal()
    frame["macd_hist"] = macd.macd_diff()

    stoch = ta.momentum.StochasticOscillator(frame["high"], frame["low"], price)
    frame["stoch"] = stoch.stoch()
    frame["stoch_signal"] = stoch.stoch_signal()

    frame["atr"] = ta.volatility.average_true_range(frame["high"], frame["low"], price)
    frame["atr_pct"] = frame["atr"] / price

    bollinger = ta.volatility.BollingerBands(price)
    frame["bb_high"] = bollinger.bollinger_hband()
    frame["bb_low"] = bollinger.bollinger_lband()
    frame["bb_width"] = bollinger.bollinger_wband()

    frame["adx"] = ta.trend.adx(frame["high"], frame["low"], price)
    frame["cci"] = ta.trend.cci(frame["high"], frame["low"], price, window=20)
    frame["roc"] = ta.momentum.roc(price, window=5)
    frame["williams_r"] = ta.momentum.williams_r(frame["high"], frame["low"], price, lbp=14)
    frame["mfi"] = ta.volume.money_flow_index(frame["high"], frame["low"], price, frame["volume"], window=14)
    frame["obv"] = ta.volume.on_balance_volume(price, frame["volume"])

    frame["body"] = (frame["close"] - frame["open"]).abs()
    frame["upper_wick"] = frame["high"] - frame[["open", "close"]].max(axis=1)
    frame["lower_wick"] = frame[["open", "close"]].min(axis=1) - frame["low"]
    frame["candle_range"] = frame["high"] - frame["low"]
    frame["body_ratio"] = frame["body"] / frame["candle_range"].replace(0, np.nan)
    frame["range_pct"] = frame["candle_range"] / price
    frame["body_pct"] = (frame["close"] - frame["open"]) / price

    frame["volume_sma20"] = frame["volume"].rolling(20).mean()
    frame["volume_ratio_20"] = frame["volume"] / frame["volume_sma20"]
    frame["volume_sma5"] = frame["volume"].rolling(5).mean()
    frame["volume_ratio_5"] = frame["volume"] / frame["volume_sma5"]

    frame["hour"] = frame["time"].dt.hour
    frame["day_of_week"] = frame["time"].dt.dayofweek

    frame["return_1"] = price.pct_change(1)
    frame["return_3"] = price.pct_change(3)
    frame["return_5"] = price.pct_change(5)
    frame["return_10"] = price.pct_change(10)
    frame["rolling_volatility_10"] = frame["return_1"].rolling(10).std()

    frame["price_vs_ma20"] = (price - frame["ma20"]) / price
    frame["price_vs_ma50"] = (price - frame["ma50"]) / price
    frame["price_vs_ma200"] = (price - frame["ma200"]) / price
    frame["ema_gap_20_50"] = (frame["ema20"] - frame["ema50"]) / price
    frame["ma_gap_50_200"] = (frame["ma50"] - frame["ma200"]) / price
    frame["trend_strength"] = (frame["ma50"] - frame["ma200"]).abs() / price
    frame["ma20_slope_5"] = frame["ma20"].pct_change(5)
    frame["ma50_slope_5"] = frame["ma50"].pct_change(5)

    frame["close_1"] = frame["close"].shift(1)
    frame["close_2"] = frame["close"].shift(2)
    frame["close_3"] = frame["close"].shift(3)
    frame["rsi_1"] = frame["rsi"].shift(1)
    frame["rsi_change"] = frame["rsi"].diff()
    frame["macd_hist_change"] = frame["macd_hist"].diff()
    frame["volume_1"] = frame["volume"].shift(1)
    frame["obv_change_5"] = frame["obv"].diff(5)

    required_columns = FEATURE_COLUMNS.copy()

    if include_target:
        frame["target"] = _build_trade_targets(frame)
        required_columns.append("target")

    frame[required_columns] = frame[required_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=required_columns).copy()
    return frame
