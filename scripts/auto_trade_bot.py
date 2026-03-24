from _bootstrap import apply_runtime_cli


apply_runtime_cli()

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
