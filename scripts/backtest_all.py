import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binance_future_prediction.backtesting import backtest_symbol
from binance_future_prediction.settings import get_provider_settings
from binance_future_prediction.universe import SYMBOLS


def main() -> None:
    provider_name, _, _ = get_provider_settings(require_credentials=False, public_only=True)
    print(f"Backtesting provider: {provider_name}")

    for symbol in SYMBOLS:
        try:
            result = backtest_symbol(symbol)
        except Exception as exc:
            print("====================")
            print("Symbol:", symbol)
            print("====================")
            print("Backtest failed:", exc)
            continue

        print("====================")
        print("Symbol:", result["symbol"])
        print("====================")
        print("Final balance:", result["final_balance"])
        print("Trades:", result["trades"])
        print("Wins:", result["wins"])
        print("Losses:", result["losses"])
        print("Win rate:", result["win_rate"])
        print("Test rows:", result["test_rows"])


if __name__ == "__main__":
    main()
