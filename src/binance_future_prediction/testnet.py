import logging

from .exchanges import create_exchange_client
from .settings import get_provider_settings


LOGGER = logging.getLogger(__name__)


def create_exchange_runtime_client(require_credentials: bool = True):
    provider_name, provider_settings, _ = get_provider_settings(require_credentials=require_credentials, public_only=False)
    client = create_exchange_client(require_credentials=require_credentials, public_only=False)
    LOGGER.info(
        "Connected to provider=%s mode=%s market_data_mode=%s endpoint=%s",
        provider_name,
        provider_settings.get("mode"),
        provider_settings.get("market_data_mode"),
        getattr(client, "endpoint", ""),
    )
    return client


def create_testnet_client():
    return create_exchange_runtime_client(require_credentials=True)
