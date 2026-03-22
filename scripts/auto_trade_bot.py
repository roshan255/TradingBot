import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binance_future_prediction.bot import AutoTradingBot
from binance_future_prediction.logging_utils import setup_logging
from binance_future_prediction.testnet import create_testnet_client


def main() -> None:
    setup_logging()
    client = create_testnet_client()
    bot = AutoTradingBot(client)
    bot.run_forever()


if __name__ == "__main__":
    main()
