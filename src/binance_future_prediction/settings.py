import json
import os
from functools import lru_cache

from .config import (
    BALANCE_USAGE_FRACTION,
    BINANCE_FUTURES_PRODUCTION_URL,
    BINANCE_FUTURES_TESTNET_URL,
    DEFAULT_LEVERAGE,
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


def _default_settings() -> dict:
    return {
        "mode": "testnet",
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
    if not settings_path.exists():
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


def _apply_env_overrides(settings: dict) -> dict:
    mode = os.environ.get("BINANCE_MODE") or os.environ.get("BFP_MODE") or settings.get("mode", "testnet")
    settings["mode"] = mode

    for mode_name, prefix, default_url in [
        ("testnet", "BINANCE_TESTNET", BINANCE_FUTURES_TESTNET_URL),
        ("production", "BINANCE_PRODUCTION", BINANCE_FUTURES_PRODUCTION_URL),
    ]:
        mode_settings = settings.setdefault(mode_name, {})
        mode_settings["api_key"] = os.environ.get(f"{prefix}_API_KEY", mode_settings.get("api_key", ""))
        mode_settings["api_secret"] = os.environ.get(f"{prefix}_API_SECRET", mode_settings.get("api_secret", ""))
        mode_settings["futures_url"] = os.environ.get(f"{prefix}_FUTURES_URL", mode_settings.get("futures_url", default_url))

    active_mode = settings["mode"]
    generic_key = os.environ.get("BINANCE_API_KEY")
    generic_secret = os.environ.get("BINANCE_API_SECRET")
    generic_url = os.environ.get("BINANCE_FUTURES_URL")
    if generic_key:
        settings.setdefault(active_mode, {})["api_key"] = generic_key
    if generic_secret:
        settings.setdefault(active_mode, {})["api_secret"] = generic_secret
    if generic_url:
        settings.setdefault(active_mode, {})["futures_url"] = generic_url

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


def _normalize_settings(settings: dict) -> dict:
    mode = settings.get("mode", "testnet")
    if mode not in {"testnet", "production"}:
        raise ValueError("mode must be 'testnet' or 'production'")

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
    settings = _default_settings()
    file_settings = _load_settings_file()
    if file_settings:
        _deep_update(settings, file_settings)
    _apply_env_overrides(settings)
    return _normalize_settings(settings)


def get_mode_settings(require_credentials: bool = False) -> tuple[str, dict, dict]:
    settings = load_runtime_settings()
    mode = settings["mode"]
    mode_settings = settings.get(mode, {})
    if require_credentials and (not mode_settings.get("api_key") or not mode_settings.get("api_secret")):
        raise ValueError(
            "Missing Binance credentials. Provide local/trading_config.json or set BINANCE_API_KEY/BINANCE_API_SECRET."
        )
    return mode, mode_settings, settings


def get_trading_settings() -> dict:
    return load_runtime_settings().get("trading", {})


def clear_runtime_settings_cache() -> None:
    load_runtime_settings.cache_clear()
