import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binance_future_prediction.config import DOWNLOAD_LIMIT, INTERVAL, TOTAL_CANDLES
from binance_future_prediction.data import ensure_historical_data
from binance_future_prediction.training import create_and_save_features, train_model
from binance_future_prediction.universe import SYMBOLS


def main() -> None:
    total_symbols = len(SYMBOLS)
    for index, symbol in enumerate(SYMBOLS, start=1):
        print("\n====================")
        print(f"Processing: {symbol} ({index}/{total_symbols})")
        print("====================")

        print(f"[{symbol}] Step 1/3: refreshing historical data...")
        ensure_historical_data(symbol, INTERVAL, DOWNLOAD_LIMIT, TOTAL_CANDLES)

        print(f"[{symbol}] Step 2/3: generating features...")
        feature_df = create_and_save_features(symbol)

        print(f"[{symbol}] Step 3/3: training and selecting best model...")
        metrics = train_model(symbol)

        print(f"[{symbol}] Completed")
        print("Features created:", symbol, "Rows:", len(feature_df))
        print("Selected model:", metrics["selected_model"], f"({metrics['selected_kind']})")
        print("Accuracy:", round(metrics["accuracy"], 4))
        print("Baseline Accuracy:", round(metrics["baseline_accuracy"], 4))
        print("Macro F1:", round(metrics["macro_f1"], 4))
        print("Balanced Accuracy:", round(metrics["balanced_accuracy"], 4))
        print("Trade Precision:", round(metrics["trade_precision"], 4))
        print("Signal Threshold:", metrics["signal_rules"]["signal_threshold"])
        print("Class Gap:", metrics["signal_rules"]["class_gap"])
        print(metrics["report"])


if __name__ == "__main__":
    main()
