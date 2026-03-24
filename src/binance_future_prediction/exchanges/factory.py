from .base import ExchangeProvider
from ..settings import get_provider_settings


def create_exchange_client(require_credentials: bool = False, public_only: bool = False) -> ExchangeProvider:
    provider_name, provider_settings, _ = get_provider_settings(require_credentials=require_credentials, public_only=public_only)
    if provider_name == "binance":
        from .binance_provider import BinanceExchangeProvider

        return BinanceExchangeProvider(provider_settings, require_credentials=require_credentials, public_only=public_only)
    if provider_name == "bybit":
        from .bybit_provider import BybitExchangeProvider

        return BybitExchangeProvider(provider_settings, require_credentials=require_credentials, public_only=public_only)
    raise ValueError(f"Unsupported provider: {provider_name}")
