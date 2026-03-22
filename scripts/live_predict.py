import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binance_future_prediction.live import generate_live_signals


def main() -> None:
    for signal in generate_live_signals():
        print("\nSymbol:", signal.symbol)
        print("Time:", signal.current_time)
        print("Current Price:", signal.price)
        print("Prediction Probability:", round(signal.probability, 3))
        print(signal.prediction)


if __name__ == "__main__":
    main()
