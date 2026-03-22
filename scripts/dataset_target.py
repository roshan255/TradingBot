import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binance_future_prediction.paths import get_symbol_file
from binance_future_prediction.universe import SYMBOLS


def main() -> None:
    for symbol in SYMBOLS:
        df = pd.read_csv(get_symbol_file(symbol, "features.csv"))
        print("Symbol:", symbol)
        print(df["target"].value_counts())


if __name__ == "__main__":
    main()
