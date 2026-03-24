import json
import os
from functools import lru_cache

from .config import (
    BALANCE_USAGE_FRACTION,
    BINANCE_FUTURES_PRODUCTION_URL,
    BINANCE_FUTURES_TESTNET_URL,
    BYBIT_FUTURES_PRODUCTION_URL,
    BYBIT_FUTURES_TESTNET_URL,
    DEFAULT_LEVERAGE,
    DEFAULT_MARKET_DATA_MODE,
    DEFAULT_PROVIDER,
    POSITION_SIZE_USDT,
    SETTINGS_FILE,
    USE_FULL_ACCOUNT_BALANCE,
)
from .paths import LOCAL_DIR, resolve_runtime_path


DEFAULT_NEWS = {
    "enabled": False,
    "provider": "rss",
    "macro_query": "crypto OR bitcoin OR fed OR inflation OR war OR sanctions OR oil OR gas",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_provider_settings() -> dict:
    return {
        "provider": DEFAULT_PROVIDER,
        "providers": {
            "binance": {
                "mode": "testnet",
                "market_data_mode": DEFAULT_MARKET_DATA_MODE,
                "testnet": {
                    "api_key": "",
                    "api_secret": "",
                    "futures_url": BINANCE_FUTURES_TESTNET_URL,
                },
                "production": {
                    "api_key": "",
                    "api_secret": "",
                    "futures_url": BINANCE_FUTURES_PRODUCTION_URL,
                },
            },
            "bybit": {
                "mode": "testnet",
                "market_data_mode": DEFAULT_MARKET_DATA_MODE,
                "category": "linear",
                "settle_coin": "USDT",
                "account_type": "UNIFIED",
                "position_idx": 0,
                "testnet": {
                    "api_key": "",
                    "api_secret": "",
                    "futures_url": BYBIT_FUTURES_TESTNET_URL,
                },
                "production": {
                    "api_key": "",
                    "api_secret": "",
                    "futures_url": BYBIT_FUTURES_PRODUCTION_URL,
                },
            },
        },
        "trading": {
            "default_leverage": DEFAULT_LEVERAGE,
            "fixed_margin_usdt": POSITION_SIZE_USDT,
            "use_full_account_balance": USE_FULL_ACCOUNT_BALANCE,
            "balance_usage_fraction": BALANCE_USAGE_FRACTION,
        },
        "news": dict(DEFAULT_NEWS),
    }


def _load_settings_file() -> dict:
    settings_file = os.environ.get("BFP_SETTINGS_FILE", SETTINGS_FILE)
    settings_path = resolve_runtime_path(settings_file, base_dir=LOCAL_DIR.parent)
    if not settings_path.exists() or settings_path.is_dir():
        return {}
    with settings_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _migrate_legacy_settings(settings: dict) -> dict:
    if "providers" in settings:
        return settings

    has_legacy_binance = any(key in settings for key in ["mode", "testnet", "production"])
    if not has_legacy_binance:
        return settings

    providers = settings.setdefault("providers", {})
    providers.setdefault(
        "binance",
        {
            "mode": settings.get("mode", "testnet"),
            "market_data_mode": settings.get("market_data_mode", DEFAULT_MARKET_DATA_MODE),
            "testnet": settings.get(
                "testnet",
                {
                    "api_key": "",
                    "api_secret": "",
                    "futures_url": BINANCE_FUTURES_TESTNET_URL,
                },
            ),
            "production": settings.get(
                "production",
                {
                    "api_key": "",
                    "api_secret": "",
                    "futures_url": BINANCE_FUTURES_PRODUCTION_URL,
                },
            ),
        },
    )
    settings.setdefault("provider", "binance")
    return settings


def _apply_provider_env_overrides(provider_name: str, provider_settings: dict) -> None:
    prefix = provider_name.upper()
    provider_settings["mode"] = os.environ.get(f"{prefix}_MODE", provider_settings.get("mode", "testnet"))
    provider_settings["market_data_mode"] = os.environ.get(
        f"{prefix}_MARKET_DATA_MODE",
        provider_settings.get("market_data_mode", DEFAULT_MARKET_DATA_MODE),
    )

    for mode_name in ["testnet", "production"]:
        mode_settings = provider_settings.setdefault(mode_name, {})
        env_mode_prefix = f"{prefix}_{mode_name.upper()}"
        mode_settings["api_key"] = os.environ.get(f"{env_mode_prefix}_API_KEY", mode_settings.get("api_key", ""))
        mode_settings["api_secret"] = os.environ.get(f"{env_mode_prefix}_API_SECRET", mode_settings.get("api_secret", ""))
        mode_settings["futures_url"] = os.environ.get(
            f"{env_mode_prefix}_FUTURES_URL",
            mode_settings.get("futures_url", ""),
        )

    active_mode = provider_settings.get("mode", "testnet")
    active_mode_settings = provider_settings.setdefault(active_mode, {})
    generic_key = os.environ.get(f"{prefix}_API_KEY")
    generic_secret = os.environ.get(f"{prefix}_API_SECRET")
    generic_url = os.environ.get(f"{prefix}_FUTURES_URL")
    if generic_key:
        active_mode_settings["api_key"] = generic_key
    if generic_secret:
        active_mode_settings["api_secret"] = generic_secret
    if generic_url:
        active_mode_settings["futures_url"] = generic_url

    if provider_name == "bybit":
        provider_settings["category"] = os.environ.get("BYBIT_CATEGORY", provider_settings.get("category", "linear"))
        provider_settings["settle_coin"] = os.environ.get("BYBIT_SETTLE_COIN", provider_settings.get("settle_coin", "USDT"))
        provider_settings["account_type"] = os.environ.get("BYBIT_ACCOUNT_TYPE", provider_settings.get("account_type", "UNIFIED"))
        provider_settings["position_idx"] = int(os.environ.get("BYBIT_POSITION_IDX", provider_settings.get("position_idx", 0)))


def _apply_env_overrides(settings: dict) -> dict:
    settings["provider"] = os.environ.get("BFP_PROVIDER", settings.get("provider", DEFAULT_PROVIDER)).lower()

    providers = settings.setdefault("providers", {})
    for provider_name in ["binance", "bybit"]:
        provider_settings = providers.setdefault(provider_name, {})
        _apply_provider_env_overrides(provider_name, provider_settings)

    selected_provider = settings["provider"]
    selected_settings = providers.setdefault(selected_provider, {})
    bfp_mode = os.environ.get("BFP_MODE")
    if bfp_mode:
        selected_settings["mode"] = bfp_mode
    bfp_market_data_mode = os.environ.get("BFP_MARKET_DATA_MODE")
    if bfp_market_data_mode:
        selected_settings["market_data_mode"] = bfp_market_data_mode

    trading = settings.setdefault("trading", {})
    trading["default_leverage"] = int(os.environ.get("BFP_DEFAULT_LEVERAGE", trading.get("default_leverage", DEFAULT_LEVERAGE)))
    trading["fixed_margin_usdt"] = float(os.environ.get("BFP_FIXED_MARGIN_USDT", trading.get("fixed_margin_usdt", POSITION_SIZE_USDT)))
    trading["use_full_account_balance"] = _env_bool(
        "BFP_USE_FULL_ACCOUNT_BALANCE",
        bool(trading.get("use_full_account_balance", USE_FULL_ACCOUNT_BALANCE)),
    )
    trading["balance_usage_fraction"] = float(
        os.environ.get("BFP_BALANCE_USAGE_FRACTION", trading.get("balance_usage_fraction", BALANCE_USAGE_FRACTION))
    )

    news = settings.setdefault("news", dict(DEFAULT_NEWS))
    news["enabled"] = _env_bool("BFP_NEWS_ENABLED", bool(news.get("enabled", False)))
    news["provider"] = os.environ.get("BFP_NEWS_PROVIDER", news.get("provider", DEFAULT_NEWS["provider"]))
    news["macro_query"] = os.environ.get("BFP_MACRO_QUERY", news.get("macro_query", DEFAULT_NEWS["macro_query"]))
    return settings


def _normalize_provider_settings(provider_name: str, provider_settings: dict) -> dict:
    provider_settings["mode"] = provider_settings.get("mode", "testnet")
    provider_settings["market_data_mode"] = provider_settings.get("market_data_mode", DEFAULT_MARKET_DATA_MODE)
    if provider_settings["mode"] not in {"testnet", "production"}:
        raise ValueError(f"{provider_name} mode must be 'testnet' or 'production'")
    if provider_settings["market_data_mode"] not in {"testnet", "production"}:
        raise ValueError(f"{provider_name} market_data_mode must be 'testnet' or 'production'")

    for mode_name, default_url in [
        ("testnet", BINANCE_FUTURES_TESTNET_URL if provider_name == "binance" else BYBIT_FUTURES_TESTNET_URL),
        ("production", BINANCE_FUTURES_PRODUCTION_URL if provider_name == "binance" else BYBIT_FUTURES_PRODUCTION_URL),
    ]:
        mode_settings = provider_settings.setdefault(mode_name, {})
        mode_settings["api_key"] = mode_settings.get("api_key", "")
        mode_settings["api_secret"] = mode_settings.get("api_secret", "")
        mode_settings["futures_url"] = mode_settings.get("futures_url", default_url)

    if provider_name == "bybit":
        provider_settings["category"] = provider_settings.get("category", "linear")
        provider_settings["settle_coin"] = provider_settings.get("settle_coin", "USDT")
        provider_settings["account_type"] = provider_settings.get("account_type", "UNIFIED")
        provider_settings["position_idx"] = int(provider_settings.get("position_idx", 0))
    return provider_settings


def _normalize_settings(settings: dict) -> dict:
    provider = settings.get("provider", DEFAULT_PROVIDER).lower()
    if provider not in {"binance", "bybit"}:
        raise ValueError("provider must be 'binance' or 'bybit'")
    settings["provider"] = provider

    providers = settings.setdefault("providers", {})
    for provider_name in ["binance", "bybit"]:
        providers[provider_name] = _normalize_provider_settings(provider_name, providers.setdefault(provider_name, {}))

    settings.setdefault("news", dict(DEFAULT_NEWS))
    trading = settings.setdefault(
        "trading",
        {
            "default_leverage": DEFAULT_LEVERAGE,
            "fixed_margin_usdt": POSITION_SIZE_USDT,
            "use_full_account_balance": USE_FULL_ACCOUNT_BALANCE,
            "balance_usage_fraction": BALANCE_USAGE_FRACTION,
        },
    )
    trading["default_leverage"] = max(1, int(trading.get("default_leverage", DEFAULT_LEVERAGE)))
    trading["fixed_margin_usdt"] = max(1.0, float(trading.get("fixed_margin_usdt", POSITION_SIZE_USDT)))
    trading["use_full_account_balance"] = bool(trading.get("use_full_account_balance", USE_FULL_ACCOUNT_BALANCE))
    trading["balance_usage_fraction"] = min(
        1.0,
        max(0.05, float(trading.get("balance_usage_fraction", BALANCE_USAGE_FRACTION))),
    )
    return settings


@lru_cache(maxsize=1)
def load_runtime_settings() -> dict:
    settings = _default_provider_settings()
    file_settings = _load_settings_file()
    if file_settings:
        _deep_update(settings, _migrate_legacy_settings(file_settings))
    _apply_env_overrides(settings)
    return _normalize_settings(settings)


def get_provider_settings(require_credentials: bool = False, public_only: bool = False) -> tuple[str, dict, dict]:
    settings = load_runtime_settings()
    provider_name = settings["provider"]
    provider_settings = settings["providers"].get(provider_name, {})
    mode_key = provider_settings.get("market_data_mode") if public_only else provider_settings.get("mode")
    mode_settings = provider_settings.get(mode_key, {})
    if require_credentials and (not mode_settings.get("api_key") or not mode_settings.get("api_secret")):
        raise ValueError(
            f"Missing {provider_name} credentials for mode '{mode_key}'. Provide runtime settings or env vars first."
        )
    return provider_name, provider_settings, settings


def get_mode_settings(require_credentials: bool = False) -> tuple[str, dict, dict]:
    provider_name, provider_settings, settings = get_provider_settings(require_credentials=require_credentials, public_only=False)
    mode_settings = provider_settings.get(provider_settings.get("mode", "testnet"), {})
    if require_credentials and (not mode_settings.get("api_key") or not mode_settings.get("api_secret")):
        raise ValueError(
            f"Missing {provider_name} credentials for mode '{provider_settings.get('mode', 'testnet')}'."
        )
    return provider_settings.get("mode", "testnet"), mode_settings, settings


def get_trading_settings() -> dict:
    return load_runtime_settings().get("trading", {})


def clear_runtime_settings_cache() -> None:
    load_runtime_settings.cache_clear()
