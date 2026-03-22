import logging

from binance.client import Client

from .settings import get_mode_settings


LOGGER = logging.getLogger(__name__)


def create_testnet_client() -> Client:
    mode, mode_settings, _ = get_mode_settings(require_credentials=True)

    client = Client(
        mode_settings["api_key"],
        mode_settings["api_secret"],
        testnet=(mode == "testnet"),
    )
    client.FUTURES_URL = mode_settings["futures_url"]

    LOGGER.info("Connected to Binance Futures mode=%s url=%s", mode, mode_settings["futures_url"])
    return client
