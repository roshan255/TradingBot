from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
import logging

import joblib
import pandas as pd
from binance.client import Client

from .config import (
    INTERVAL,
    KLINES_LIMIT,
    MAX_NO_TRADE_PROBABILITY,
    MIN_ADX_FOR_TRADE,
    MIN_CLASS_GAP,
    MIN_LIVE_VOLATILITY,
    SIGNAL_THRESHOLD,
    USE_TREND_FILTER,
)
from .context import context_blocks_trade, get_market_context
from .data import get_latest_klines
from .features import create_feature_frame
from .paths import get_symbol_file
from .universe import FEATURE_COLUMNS, SYMBOLS


LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def load_model(symbol: str):
    return joblib.load(get_symbol_file(symbol, "model.pkl"))


@lru_cache(maxsize=256)
def load_model_meta(symbol: str) -> dict:
    meta_path = get_symbol_file(symbol, "model_meta.json")
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class PredictionCandidate:
    symbol: str
    side: str
    probability: float
    second_best_probability: float
    no_trade_probability: float
    price: float
    current_time: str
    adx: float
    signal_threshold: float
    class_gap_threshold: float
    context_score: float
    context_headline: str


@dataclass
class PredictionEvaluation:
    symbol: str
    candidate: PredictionCandidate | None
    rejection_reason: str | None
    best_side: str
    best_probability: float
    second_best_probability: float
    no_trade_probability: float
    adx: float
    price: float
    context_score: float


def evaluate_symbol(client: Client, symbol: str) -> PredictionEvaluation:
    try:
        model = load_model(symbol)
        metadata = load_model_meta(symbol)
        latest_df = get_latest_klines(symbol, INTERVAL, limit=KLINES_LIMIT, client=client)
        feature_df = create_feature_frame(latest_df, include_target=False)
    except Exception as exc:
        LOGGER.exception("Prediction setup failed for %s: %s", symbol, exc)
        return PredictionEvaluation(symbol, None, "prediction_setup_failed", "NONE", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if feature_df.empty:
        return PredictionEvaluation(symbol, None, "not_enough_candles", "NONE", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    latest = feature_df.iloc[-1:]
    volatility = latest["atr"].iat[0] / latest["close"].iat[0]
    adx = float(latest["adx"].iat[0])
    price = float(latest["close"].iat[0])
    context = get_market_context(symbol)

    signal_rules = metadata.get("signal_rules", {})
    signal_threshold = float(signal_rules.get("signal_threshold", SIGNAL_THRESHOLD))
    class_gap = float(signal_rules.get("class_gap", MIN_CLASS_GAP))

    feature_row = latest[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    probs = model.predict_proba(feature_row)[0]
    prob_dict = dict(zip(model.classes_, probs))

    short_prob = float(prob_dict.get(-1, 0.0))
    no_move_prob = float(prob_dict.get(0, 0.0))
    long_prob = float(prob_dict.get(1, 0.0))

    if long_prob >= short_prob:
        best_side = "BUY"
        best_prob = long_prob
        opposite_prob = short_prob
    else:
        best_side = "SELL"
        best_prob = short_prob
        opposite_prob = long_prob

    second_best_prob = max(opposite_prob, no_move_prob)
    rejection_reason = None
    if volatility < MIN_LIVE_VOLATILITY:
        rejection_reason = "low_volatility"
    elif adx < MIN_ADX_FOR_TRADE:
        rejection_reason = "low_adx"
    elif best_prob < signal_threshold:
        rejection_reason = "probability_below_threshold"
    elif (best_prob - opposite_prob) < class_gap:
        rejection_reason = "class_gap_too_small"
    elif no_move_prob > MAX_NO_TRADE_PROBABILITY and (best_prob - no_move_prob) < 0.03:
        rejection_reason = "no_trade_probability_too_high"
    elif no_move_prob > best_prob and (no_move_prob - best_prob) > 0.08:
        rejection_reason = "model_prefers_no_trade"
    elif context_blocks_trade(best_side, context):
        rejection_reason = "market_context_blocked_trade"
    elif USE_TREND_FILTER:
        ma50 = latest["ma50"].iat[0]
        ma200 = latest["ma200"].iat[0]
        price_vs_ma50 = float(latest["price_vs_ma50"].iat[0])
        if best_side == "BUY" and (ma50 < ma200 or price_vs_ma50 < 0):
            rejection_reason = "trend_filter_blocked_long"
        elif best_side == "SELL" and (ma50 > ma200 or price_vs_ma50 > 0):
            rejection_reason = "trend_filter_blocked_short"

    if rejection_reason is not None:
        return PredictionEvaluation(
            symbol=symbol,
            candidate=None,
            rejection_reason=rejection_reason,
            best_side=best_side,
            best_probability=best_prob,
            second_best_probability=second_best_prob,
            no_trade_probability=no_move_prob,
            adx=adx,
            price=price,
            context_score=context.score,
        )

    candidate = PredictionCandidate(
        symbol=symbol,
        side=best_side,
        probability=best_prob,
        second_best_probability=second_best_prob,
        no_trade_probability=no_move_prob,
        price=price,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        adx=adx,
        signal_threshold=signal_threshold,
        class_gap_threshold=class_gap,
        context_score=context.score,
        context_headline=context.headline,
    )
    return PredictionEvaluation(
        symbol=symbol,
        candidate=candidate,
        rejection_reason=None,
        best_side=best_side,
        best_probability=best_prob,
        second_best_probability=second_best_prob,
        no_trade_probability=no_move_prob,
        adx=adx,
        price=price,
        context_score=context.score,
    )


def predict_symbol(client: Client, symbol: str) -> PredictionCandidate | None:
    return evaluate_symbol(client, symbol).candidate


def find_best_trade(client: Client) -> PredictionCandidate | None:
    best_candidate: PredictionCandidate | None = None
    rejection_counts: dict[str, int] = {}
    strongest_rejected: PredictionEvaluation | None = None

    for symbol in SYMBOLS:
        evaluation = evaluate_symbol(client, symbol)
        if evaluation.candidate is not None:
            candidate = evaluation.candidate
            LOGGER.info(
                "Candidate %s side=%s prob=%.3f gap=%.3f no_trade=%.3f adx=%.2f context=%.2f price=%.6f",
                candidate.symbol,
                candidate.side,
                candidate.probability,
                candidate.probability - candidate.second_best_probability,
                candidate.no_trade_probability,
                candidate.adx,
                candidate.context_score,
                candidate.price,
            )
            if best_candidate is None or candidate.probability > best_candidate.probability:
                best_candidate = candidate
            continue

        reason = evaluation.rejection_reason or "unknown"
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        if strongest_rejected is None or evaluation.best_probability > strongest_rejected.best_probability:
            strongest_rejected = evaluation

    if best_candidate is None:
        if strongest_rejected is not None:
            LOGGER.info(
                "No valid trade found. Strongest rejected=%s side=%s prob=%.3f gap=%.3f no_trade=%.3f adx=%.2f context=%.2f reason=%s",
                strongest_rejected.symbol,
                strongest_rejected.best_side,
                strongest_rejected.best_probability,
                strongest_rejected.best_probability - strongest_rejected.second_best_probability,
                strongest_rejected.no_trade_probability,
                strongest_rejected.adx,
                strongest_rejected.context_score,
                strongest_rejected.rejection_reason,
            )
        if rejection_counts:
            LOGGER.info("Rejection summary: %s", rejection_counts)

    return best_candidate
