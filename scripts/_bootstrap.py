import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def apply_runtime_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--provider", choices=["binance", "bybit"])
    parser.add_argument("--mode", choices=["testnet", "production"])
    parser.add_argument("--market-data-mode", choices=["testnet", "production"])
    parser.add_argument("--storage-dir")
    parser.add_argument("--settings-file")
    args, remaining = parser.parse_known_args()

    if args.provider:
        os.environ["BFP_PROVIDER"] = args.provider
    if args.mode:
        os.environ["BFP_MODE"] = args.mode
    if args.market_data_mode:
        os.environ["BFP_MARKET_DATA_MODE"] = args.market_data_mode
    if args.storage_dir:
        os.environ["BFP_STORAGE_DIR"] = args.storage_dir
    if args.settings_file:
        os.environ["BFP_SETTINGS_FILE"] = args.settings_file

    sys.argv = [sys.argv[0], *remaining]
    return args
