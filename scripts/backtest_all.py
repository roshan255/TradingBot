import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binance_future_prediction.backtesting import backtest_symbol
from binance_future_prediction.universe import SYMBOLS


def main() -> None:
    for symbol in SYMBOLS:
        result = backtest_symbol(symbol)

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
