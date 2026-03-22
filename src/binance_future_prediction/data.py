import time

import pandas as pd
from binance.client import Client

from .paths import get_symbol_file, write_csv_atomic


def _create_client() -> Client:
    return Client()


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


def _format_klines(klines) -> pd.DataFrame:
    df = pd.DataFrame(klines)
    if df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = df[[0, 1, 2, 3, 4, 5]]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms")

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = df[column].astype(float)

    return df


def _save_symbol_data(symbol: str, df: pd.DataFrame, total_candles: int) -> pd.DataFrame:
    trimmed = df.sort_values("time").drop_duplicates(subset="time", keep="last").tail(total_candles).reset_index(drop=True)
    write_csv_atomic(get_symbol_file(symbol, "data.csv"), trimmed)
    return trimmed


def download_historical_data(symbol: str, interval: str, limit: int, total_candles: int) -> pd.DataFrame:
    client = _create_client()
    all_data = []
    end_time = None

    while len(all_data) < total_candles:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            endTime=end_time,
        )

        if not klines:
            break

        all_data = klines + all_data
        end_time = klines[0][0] - 1

        print(f"Downloading {symbol}: {len(all_data)}/{total_candles}", end="\r", flush=True)
        time.sleep(0.15)

    df = _format_klines(all_data)
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

    new_rows = []
    while True:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_time_ms,
        )

        if not klines:
            break

        batch = _format_klines(klines)
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


def get_latest_klines(symbol: str, interval: str, limit: int = 500, client: Client | None = None) -> pd.DataFrame:
    active_client = client or _create_client()
    klines = active_client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    return _format_klines(klines)
