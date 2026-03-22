import json
import os
from pathlib import Path
import time

from .paths import MPLCONFIG_DIR
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from .config import (
    DIRECTION_TOP_FEATURES,
    MULTICLASS_TOP_FEATURES,
    TRAIN_TEST_RATIO,
    TRAINING_WINDOW_CANDLES,
    TRADE_TOP_FEATURES,
    VALIDATION_RATIO,
)
from .encoded_model import EncodedClassifierModel
from .features import create_feature_frame
from .paths import dump_joblib_atomic, get_symbol_file, write_csv_atomic, write_json_atomic
from .signal_model import BlendedSignalModel, ConstantProbabilityModel, TwoStageSignalModel
from .trade_simulation import simulate_signal_strategy
from .universe import FEATURE_COLUMNS


LABELS = [-1, 0, 1]
LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}


def create_and_save_features(symbol: str) -> pd.DataFrame:
    print(f"[{symbol}] Loading raw candles...")
    raw_df = pd.read_csv(get_symbol_file(symbol, "data.csv"))
    print(f"[{symbol}] Creating features from {len(raw_df)} candles...")
    start_time = time.perf_counter()
    feature_df = create_feature_frame(raw_df, include_target=True)
    elapsed = time.perf_counter() - start_time
    feature_path = get_symbol_file(symbol, "features.csv")
    temp_path = feature_path.with_name(f"{feature_path.name}.tmp")
    feature_df.to_csv(temp_path, index=False)
    os.replace(temp_path, feature_path)
    print(f"[{symbol}] Features ready: {len(feature_df)} rows, {len(feature_df.columns)} columns in {elapsed:.1f}s")
    return feature_df


def get_split_index(df: pd.DataFrame) -> int:
    return int(len(df) * TRAIN_TEST_RATIO)


def _make_binary_model(iterations: int = 280) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=iterations,
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=4.0,
        random_strength=0.8,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        border_count=128,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def _make_multiclass_model(iterations: int = 320) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_strength=0.8,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        border_count=128,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def _recency_weights(length: int) -> np.ndarray:
    return np.linspace(0.75, 1.35, length)


def _binary_sample_weights(target: pd.Series) -> np.ndarray:
    if target.nunique() < 2:
        class_weights = np.ones(len(target), dtype=float)
    else:
        class_weights = compute_sample_weight(class_weight="balanced", y=target)
    return class_weights * _recency_weights(len(target))


def _multiclass_sample_weights(target: pd.Series) -> np.ndarray:
    if target.nunique() < 2:
        class_weights = np.ones(len(target), dtype=float)
    else:
        class_weights = compute_sample_weight(class_weight="balanced", y=target)
    return class_weights * _recency_weights(len(target))


def _fit_binary_model(name: str, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame | None, y_validation: pd.Series | None, iterations: int = 280):
    unique_classes = sorted(y_train.unique().tolist())
    if len(unique_classes) == 0:
        print(f"[{name}] No rows available, using neutral constant model")
        return ConstantProbabilityModel([0, 1], [0.5, 0.5]), 0
    if len(unique_classes) == 1:
        only_class = int(unique_classes[0])
        probabilities = [1.0, 0.0] if only_class == 0 else [0.0, 1.0]
        print(f"[{name}] Only one class present ({only_class}), using constant model")
        return ConstantProbabilityModel([0, 1], probabilities), 0

    model = _make_binary_model(iterations=iterations)
    fit_kwargs = {"sample_weight": _binary_sample_weights(y_train)}
    if X_validation is not None and y_validation is not None and len(X_validation) and y_validation.nunique() >= 2:
        fit_kwargs["eval_set"] = (X_validation, y_validation)
        fit_kwargs["use_best_model"] = True
        fit_kwargs["early_stopping_rounds"] = 30

    start = time.perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.perf_counter() - start
    best_rounds = max(80, int(model.tree_count_))
    print(f"[{name}] Fit complete in {elapsed:.1f}s using {best_rounds} trees")
    return model, best_rounds


def _fit_multiclass_model(name: str, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame | None, y_validation: pd.Series | None, iterations: int = 320):
    unique_classes = sorted(y_train.unique().tolist())
    if len(unique_classes) < 2:
        only_class = int(unique_classes[0]) if unique_classes else 1
        probabilities = np.zeros(3, dtype=float)
        probabilities[only_class] = 1.0
        print(f"[{name}] Not enough class variation, using constant model")
        return ConstantProbabilityModel(LABELS, probabilities), 0

    model = _make_multiclass_model(iterations=iterations)
    fit_kwargs = {"sample_weight": _multiclass_sample_weights(y_train)}
    if X_validation is not None and y_validation is not None and len(X_validation) and y_validation.nunique() >= 2:
        fit_kwargs["eval_set"] = (X_validation, y_validation)
        fit_kwargs["use_best_model"] = True
        fit_kwargs["early_stopping_rounds"] = 35

    start = time.perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.perf_counter() - start
    best_rounds = max(100, int(model.tree_count_))
    print(f"[{name}] Fit complete in {elapsed:.1f}s using {best_rounds} trees")
    return model, best_rounds


def _get_feature_importances(model, feature_names: list[str]) -> np.ndarray:
    if isinstance(model, ConstantProbabilityModel):
        return np.ones(len(feature_names), dtype=float)
    if hasattr(model, "get_feature_importance"):
        return np.asarray(model.get_feature_importance(), dtype=float)
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    return np.ones(len(feature_names), dtype=float)


def _select_top_features(model, feature_names: list[str], limit: int) -> list[str]:
    if limit >= len(feature_names):
        return list(feature_names)
    importances = _get_feature_importances(model, feature_names)
    order = np.argsort(importances)[::-1]
    selected = [feature_names[index] for index in order[:limit] if importances[index] > 0]
    if len(selected) < max(8, min(limit, len(feature_names)) // 2):
        selected = [feature_names[index] for index in order[:limit]]
    return selected or list(feature_names)


def _trade_precision(y_true: pd.Series, predictions: np.ndarray) -> float:
    trade_mask = predictions != 0
    if not trade_mask.any():
        return 0.0
    y_true_values = y_true.to_numpy()
    return float((predictions[trade_mask] == y_true_values[trade_mask]).mean())


def _score_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict:
    macro_f1 = f1_score(y_true, predictions, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, predictions)
    trade_precision = _trade_precision(y_true, predictions)
    score = (macro_f1 * 0.20) + (balanced_acc * 0.20) + (trade_precision * 0.60)
    return {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "trade_precision": float(trade_precision),
        "score": float(score),
    }


def _tune_signal_rules(probabilities: np.ndarray, validation_frame: pd.DataFrame, classes: np.ndarray, symbol: str, label: str) -> dict:
    print(f"[{symbol}] Tuning signal thresholds for {label}...")
    best_result = {
        "signal_threshold": 0.40,
        "class_gap": 0.02,
        "net_profit": float("-inf"),
        "trades": 0,
        "win_rate": 0.0,
        "coverage": 0.0,
        "avg_profit": 0.0,
        "score": float("-inf"),
    }
    minimum_trade_count = max(8, int(len(validation_frame) * 0.008))

    for threshold in [0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56]:
        for gap in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
            result = simulate_signal_strategy(validation_frame, probabilities, classes, threshold, gap)
            if result["trades"] < minimum_trade_count:
                continue
            coverage_score = min(1.0, result["coverage"] * 20.0)
            expectancy_score = max(-1.0, min(1.0, result["avg_profit"] / 0.40))
            score = (
                (result["win_rate"] * 0.45)
                + (expectancy_score * 0.25)
                + (coverage_score * 0.15)
                + (min(1.0, result["trades"] / max(minimum_trade_count * 2, 1)) * 0.15)
            )
            if score > best_result["score"]:
                best_result = {
                    "signal_threshold": float(threshold),
                    "class_gap": float(gap),
                    **result,
                    "score": float(score),
                }

    print(
        f"[{symbol}] Best {label} rules: threshold={best_result['signal_threshold']:.2f}, gap={best_result['class_gap']:.2f}, "
        f"trades={best_result['trades']}, win_rate={best_result['win_rate']:.3f}, net_profit={best_result['net_profit']:.2f}"
    )
    return best_result


def _candidate_score(validation_accuracy: float, validation_metrics: dict, validation_backtest: dict) -> float:
    coverage_bonus = min(1.0, validation_backtest["coverage"] * 20.0)
    return float(
        (validation_accuracy * 0.55)
        + (validation_metrics["trade_precision"] * 0.15)
        + (validation_metrics["macro_f1"] * 0.10)
        + (validation_backtest["win_rate"] * 0.15)
        + (coverage_bonus * 0.05)
    )


def _prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _majority_class_accuracy(y_true: pd.Series) -> float:
    if y_true.empty:
        return 0.0
    return float(y_true.value_counts(normalize=True).max())


def _build_two_stage_candidate(symbol: str, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series, validation_frame: pd.DataFrame) -> dict:
    trade_target_train = (y_train != 0).astype(int)
    trade_target_validation = (y_validation != 0).astype(int)
    direction_mask_train = y_train != 0
    direction_mask_validation = y_validation != 0
    direction_target_train = (y_train[direction_mask_train] == 1).astype(int)
    direction_target_validation = (y_validation[direction_mask_validation] == 1).astype(int)

    print(f"[{symbol}] Two-stage trade filter positives={int(trade_target_train.sum())} negatives={int((trade_target_train == 0).sum())}")
    trade_model_full, _ = _fit_binary_model(
        f"{symbol}:trade_filter_full",
        X_train,
        trade_target_train,
        X_validation,
        trade_target_validation,
        iterations=280,
    )
    trade_features = _select_top_features(trade_model_full, list(X_train.columns), TRADE_TOP_FEATURES)
    trade_model, trade_rounds = _fit_binary_model(
        f"{symbol}:trade_filter_selected",
        X_train.loc[:, trade_features],
        trade_target_train,
        X_validation.loc[:, trade_features],
        trade_target_validation,
        iterations=240,
    )

    print(f"[{symbol}] Two-stage direction rows={int(direction_mask_train.sum())} longs={int(direction_target_train.sum())} shorts={int((direction_target_train == 0).sum())}")
    direction_model_full, _ = _fit_binary_model(
        f"{symbol}:direction_full",
        X_train.loc[direction_mask_train],
        direction_target_train,
        X_validation.loc[direction_mask_validation],
        direction_target_validation,
        iterations=240,
    )
    direction_features = _select_top_features(direction_model_full, list(X_train.columns), DIRECTION_TOP_FEATURES)
    direction_model, direction_rounds = _fit_binary_model(
        f"{symbol}:direction_selected",
        X_train.loc[direction_mask_train, direction_features],
        direction_target_train,
        X_validation.loc[direction_mask_validation, direction_features],
        direction_target_validation,
        iterations=220,
    )

    candidate_model = TwoStageSignalModel(trade_model, direction_model, trade_features, direction_features)
    validation_predictions = candidate_model.predict(X_validation)
    validation_probabilities = candidate_model.predict_proba(X_validation)
    validation_accuracy = accuracy_score(y_validation, validation_predictions)
    validation_metrics = _score_predictions(y_validation, validation_predictions)
    print(f"[{symbol}] Two-stage validation | accuracy={validation_accuracy:.4f} trade_precision={validation_metrics['trade_precision']:.4f} macro_f1={validation_metrics['macro_f1']:.4f}")

    signal_rules = _tune_signal_rules(validation_probabilities, validation_frame, candidate_model.classes_, symbol, "two_stage")
    validation_backtest = simulate_signal_strategy(validation_frame, validation_probabilities, candidate_model.classes_, signal_rules["signal_threshold"], signal_rules["class_gap"])

    return {
        "name": "catboost_two_stage",
        "kind": "two_stage",
        "validation_model": candidate_model,
        "validation_accuracy": float(validation_accuracy),
        "validation_metrics": validation_metrics,
        "signal_rules": signal_rules,
        "validation_backtest": validation_backtest,
        "score": _candidate_score(validation_accuracy, validation_metrics, validation_backtest),
        "trade_features": trade_features,
        "direction_features": direction_features,
        "trade_iterations": max(80, trade_rounds),
        "direction_iterations": max(80, direction_rounds),
    }


def _build_multiclass_candidate(symbol: str, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series, validation_frame: pd.DataFrame) -> dict:
    y_train_encoded = y_train.map(LABEL_TO_INDEX).astype(int)
    y_validation_encoded = y_validation.map(LABEL_TO_INDEX).astype(int)

    print(f"[{symbol}] Multiclass training classes: {sorted(y_train.unique().tolist())}")
    multiclass_full, _ = _fit_multiclass_model(
        f"{symbol}:multiclass_full",
        X_train,
        y_train_encoded,
        X_validation,
        y_validation_encoded,
        iterations=320,
    )
    selected_features = _select_top_features(multiclass_full, list(X_train.columns), MULTICLASS_TOP_FEATURES)

    if isinstance(multiclass_full, ConstantProbabilityModel):
        candidate_model = multiclass_full
        multiclass_rounds = 0
    else:
        selected_model, multiclass_rounds = _fit_multiclass_model(
            f"{symbol}:multiclass_selected",
            X_train.loc[:, selected_features],
            y_train_encoded,
            X_validation.loc[:, selected_features],
            y_validation_encoded,
            iterations=260,
        )
        candidate_model = EncodedClassifierModel(selected_model, LABELS, selected_features)

    validation_predictions = candidate_model.predict(X_validation)
    validation_probabilities = candidate_model.predict_proba(X_validation)
    validation_accuracy = accuracy_score(y_validation, validation_predictions)
    validation_metrics = _score_predictions(y_validation, validation_predictions)
    print(f"[{symbol}] Multiclass validation | accuracy={validation_accuracy:.4f} trade_precision={validation_metrics['trade_precision']:.4f} macro_f1={validation_metrics['macro_f1']:.4f}")

    signal_rules = _tune_signal_rules(validation_probabilities, validation_frame, candidate_model.classes_, symbol, "multiclass")
    validation_backtest = simulate_signal_strategy(validation_frame, validation_probabilities, candidate_model.classes_, signal_rules["signal_threshold"], signal_rules["class_gap"])

    return {
        "name": "catboost_multiclass",
        "kind": "multiclass",
        "validation_model": candidate_model,
        "validation_accuracy": float(validation_accuracy),
        "validation_metrics": validation_metrics,
        "signal_rules": signal_rules,
        "validation_backtest": validation_backtest,
        "score": _candidate_score(validation_accuracy, validation_metrics, validation_backtest),
        "feature_columns": selected_features,
        "multiclass_iterations": max(100, multiclass_rounds),
    }


def _build_blend_candidate(symbol: str, two_stage_candidate: dict, multiclass_candidate: dict, X_validation: pd.DataFrame, y_validation: pd.Series, validation_frame: pd.DataFrame) -> dict:
    best_candidate = None
    for multiclass_weight in [0.55, 0.65, 0.75]:
        blend_model = BlendedSignalModel(
            [two_stage_candidate["validation_model"], multiclass_candidate["validation_model"]],
            [1.0 - multiclass_weight, multiclass_weight],
        )
        validation_predictions = blend_model.predict(X_validation)
        validation_probabilities = blend_model.predict_proba(X_validation)
        validation_accuracy = accuracy_score(y_validation, validation_predictions)
        validation_metrics = _score_predictions(y_validation, validation_predictions)
        label = f"blend_{multiclass_weight:.2f}"
        print(f"[{symbol}] Blend validation ({multiclass_weight:.2f} multiclass) | accuracy={validation_accuracy:.4f} trade_precision={validation_metrics['trade_precision']:.4f} macro_f1={validation_metrics['macro_f1']:.4f}")
        signal_rules = _tune_signal_rules(validation_probabilities, validation_frame, blend_model.classes_, symbol, label)
        validation_backtest = simulate_signal_strategy(validation_frame, validation_probabilities, blend_model.classes_, signal_rules["signal_threshold"], signal_rules["class_gap"])
        candidate = {
            "name": "catboost_blend",
            "kind": "blend",
            "validation_model": blend_model,
            "validation_accuracy": float(validation_accuracy),
            "validation_metrics": validation_metrics,
            "signal_rules": signal_rules,
            "validation_backtest": validation_backtest,
            "score": _candidate_score(validation_accuracy, validation_metrics, validation_backtest),
            "weights": [1.0 - multiclass_weight, multiclass_weight],
            "two_stage_candidate": two_stage_candidate,
            "multiclass_candidate": multiclass_candidate,
        }
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate
    return best_candidate


def _refit_final_model(candidate: dict, symbol: str, X_train_validation: pd.DataFrame, y_train_validation: pd.Series):
    if candidate["kind"] == "two_stage":
        trade_target = (y_train_validation != 0).astype(int)
        direction_mask = y_train_validation != 0
        direction_target = (y_train_validation[direction_mask] == 1).astype(int)
        final_trade_model, _ = _fit_binary_model(
            f"{symbol}:trade_filter_final",
            X_train_validation.loc[:, candidate["trade_features"]],
            trade_target,
            None,
            None,
            iterations=candidate["trade_iterations"],
        )
        final_direction_model, _ = _fit_binary_model(
            f"{symbol}:direction_final",
            X_train_validation.loc[direction_mask, candidate["direction_features"]],
            direction_target,
            None,
            None,
            iterations=candidate["direction_iterations"],
        )
        return TwoStageSignalModel(final_trade_model, final_direction_model, candidate["trade_features"], candidate["direction_features"])

    if candidate["kind"] == "multiclass":
        y_encoded = y_train_validation.map(LABEL_TO_INDEX).astype(int)
        feature_columns = candidate["feature_columns"]
        final_multiclass_model, _ = _fit_multiclass_model(
            f"{symbol}:multiclass_final",
            X_train_validation.loc[:, feature_columns],
            y_encoded,
            None,
            None,
            iterations=candidate["multiclass_iterations"],
        )
        if isinstance(final_multiclass_model, ConstantProbabilityModel):
            return final_multiclass_model
        return EncodedClassifierModel(final_multiclass_model, LABELS, feature_columns)

    final_two_stage_model = _refit_final_model(candidate["two_stage_candidate"], symbol, X_train_validation, y_train_validation)
    final_multiclass_model = _refit_final_model(candidate["multiclass_candidate"], symbol, X_train_validation, y_train_validation)
    return BlendedSignalModel([final_two_stage_model, final_multiclass_model], candidate["weights"])


def train_model(symbol: str) -> dict:
    print(f"[{symbol}] Loading feature dataset...")
    full_df = pd.read_csv(get_symbol_file(symbol, "features.csv"))
    df = full_df.tail(TRAINING_WINDOW_CANDLES).reset_index(drop=True).copy()
    print(f"[{symbol}] Using recent window: {len(df)} rows from {len(full_df)} total")

    X = _prepare_feature_matrix(df)
    y = df["target"].astype(int)

    train_end = int(len(df) * (TRAIN_TEST_RATIO - VALIDATION_RATIO))
    validation_end = int(len(df) * TRAIN_TEST_RATIO)
    print(f"[{symbol}] Split sizes -> train: {train_end}, validation: {validation_end - train_end}, test: {len(df) - validation_end}")

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_validation = X.iloc[train_end:validation_end]
    y_validation = y.iloc[train_end:validation_end]
    validation_frame = df.iloc[train_end:validation_end].reset_index(drop=True)
    X_train_validation = X.iloc[:validation_end]
    y_train_validation = y.iloc[:validation_end]
    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]
    test_frame = df.iloc[validation_end:].reset_index(drop=True)

    print(
        f"[{symbol}] Test class mix | short={int((y_test == -1).sum())} no_trade={int((y_test == 0).sum())} long={int((y_test == 1).sum())} baseline_accuracy={_majority_class_accuracy(y_test):.4f}"
    )

    two_stage_candidate = _build_two_stage_candidate(symbol, X_train, y_train, X_validation, y_validation, validation_frame)
    multiclass_candidate = _build_multiclass_candidate(symbol, X_train, y_train, X_validation, y_validation, validation_frame)
    blend_candidate = _build_blend_candidate(symbol, two_stage_candidate, multiclass_candidate, X_validation, y_validation, validation_frame)

    candidates = [two_stage_candidate, multiclass_candidate, blend_candidate]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best_candidate = candidates[0]

    print(f"[{symbol}] Selected model {best_candidate['name']} | validation_accuracy={best_candidate['validation_accuracy']:.4f} trade_precision={best_candidate['validation_metrics']['trade_precision']:.4f} score={best_candidate['score']:.4f}")

    print(f"[{symbol}] Refitting final selected model on train+validation...")
    final_model = _refit_final_model(best_candidate, symbol, X_train_validation, y_train_validation)

    print(f"[{symbol}] Running final test evaluation...")
    predictions = final_model.predict(X_test)
    probabilities_test = final_model.predict_proba(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    test_metrics = _score_predictions(y_test, predictions)
    majority_accuracy = _majority_class_accuracy(y_test)
    test_backtest = simulate_signal_strategy(test_frame, probabilities_test, final_model.classes_, best_candidate["signal_rules"]["signal_threshold"], best_candidate["signal_rules"]["class_gap"])

    dump_joblib_atomic(get_symbol_file(symbol, "model.pkl"), final_model)
    metadata = {
        "symbol": symbol,
        "rows": len(df),
        "training_window_rows": len(df),
        "feature_count": len(FEATURE_COLUMNS),
        "selected_model": best_candidate["name"],
        "selected_kind": best_candidate["kind"],
        "signal_rules": best_candidate["signal_rules"],
        "validation_accuracy": best_candidate["validation_accuracy"],
        "validation_metrics": best_candidate["validation_metrics"],
        "validation_backtest": best_candidate["validation_backtest"],
        "test_accuracy": float(test_accuracy),
        "test_baseline_accuracy": float(majority_accuracy),
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_trade_precision": test_metrics["trade_precision"],
        "test_backtest": test_backtest,
    }
    write_json_atomic(get_symbol_file(symbol, "model_meta.json"), metadata)

    print(f"[{symbol}] Saved model + metadata | accuracy={test_accuracy:.4f} baseline={majority_accuracy:.4f} trade_precision={test_metrics['trade_precision']:.4f} test_win_rate={test_backtest['win_rate']:.3f} test_trades={test_backtest['trades']}")

    return {
        "symbol": symbol,
        "rows": len(df),
        "split_index": validation_end,
        "accuracy": float(test_accuracy),
        "baseline_accuracy": float(majority_accuracy),
        "macro_f1": test_metrics["macro_f1"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "trade_precision": test_metrics["trade_precision"],
        "selected_model": best_candidate["name"],
        "selected_kind": best_candidate["kind"],
        "signal_rules": best_candidate["signal_rules"],
        "candidate_scores": {candidate["name"]: candidate["score"] for candidate in candidates},
        "report": classification_report(y_test, predictions, labels=LABELS, zero_division=0),
        "test_backtest": test_backtest,
    }


