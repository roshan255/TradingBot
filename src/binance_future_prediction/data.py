import time

import pandas as pd

from .exchanges import ExchangeProvider, create_exchange_client
from .paths import get_symbol_file, write_csv_atomic


def _create_client() -> ExchangeProvider:
    return create_exchange_client(require_credentials=False, public_only=True)


def _read_cached_data(symbol: str) -> pd.DataFrame | None:
    path = get_symbol_file(symbol, "data.csv")
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"])
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = df[column].astype(float)
    return df


def _save_symbol_data(symbol: str, df: pd.DataFrame, total_candles: int) -> pd.DataFrame:
    trimmed = df.sort_values("time").drop_duplicates(subset="time", keep="last").tail(total_candles).reset_index(drop=True)
    write_csv_atomic(get_symbol_file(symbol, "data.csv"), trimmed)
    return trimmed


def download_historical_data(symbol: str, interval: str, limit: int, total_candles: int) -> pd.DataFrame:
    client = _create_client()
    all_batches: list[pd.DataFrame] = []
    end_time = None
    fetched = 0

    while fetched < total_candles:
        batch = client.fetch_klines(symbol, interval, limit=limit, end_time_ms=end_time)
        if batch.empty:
            break

        all_batches.insert(0, batch)
        fetched = sum(len(frame) for frame in all_batches)
        end_time = int(batch["time"].iloc[0].timestamp() * 1000) - 1

        print(f"Downloading {symbol}: {fetched}/{total_candles}", end="\r", flush=True)
        if len(batch) < limit:
            break
        time.sleep(0.15)

    if not all_batches:
        raise RuntimeError(f"No historical data fetched for {symbol}")

    df = pd.concat(all_batches, ignore_index=True)
    return _save_symbol_data(symbol, df, total_candles)


def ensure_historical_data(symbol: str, interval: str, limit: int, total_candles: int) -> pd.DataFrame:
    cached = _read_cached_data(symbol)
    client = _create_client()

    if cached is None:
        print(f"No cached data for {symbol}. Starting initial download...")
        return download_historical_data(symbol, interval, limit, total_candles)

    print(f"Using cached data for {symbol}: {len(cached)} candles")
    last_time = cached["time"].iloc[-1]
    start_time_ms = int(last_time.timestamp() * 1000) + 1

    new_rows: list[pd.DataFrame] = []
    while True:
        batch = client.fetch_klines(symbol, interval, limit=limit, start_time_ms=start_time_ms)
        if batch.empty:
            break

        new_rows.append(batch)
        start_time_ms = int(batch["time"].iloc[-1].timestamp() * 1000) + 1

        print(f"Refreshing {symbol}: +{sum(len(frame) for frame in new_rows)} candles", end="\r", flush=True)
        if len(batch) < limit:
            break
        time.sleep(0.10)

    if new_rows:
        combined = pd.concat([cached, *new_rows], ignore_index=True)
        result = _save_symbol_data(symbol, combined, total_candles)
        print(f"Refreshed {symbol}: {len(result)} cached candles total")
        return result

    trimmed = _save_symbol_data(symbol, cached, total_candles)
    print(f"No new candles for {symbol}. Reusing {len(trimmed)} cached candles")
    return trimmed


def get_latest_klines(symbol: str, interval: str, limit: int = 500, client: ExchangeProvider | None = None) -> pd.DataFrame:
    active_client = client or _create_client()
    return active_client.fetch_klines(symbol, interval, limit=limit)
