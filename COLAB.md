# Run Anywhere Guide

This project supports multiple exchange providers and can run locally, in Google Colab, or on cloud VMs like Oracle Cloud, AWS, Hetzner, or DigitalOcean.

## Supported providers
- Binance Futures
- Bybit Linear Futures

The runtime can be selected with:
- `local/trading_config.json`
- environment variables
- direct script flags like `--provider bybit`

Environment variables override file settings. Script flags override both for that run.

## 1. Install dependencies

In Google Colab, use `%cd` instead of `!cd`.
`!cd` only changes directory for that one shell command and does not persist to the next cell.

```python
!git clone <your-repo-url>
%cd /content/BinanceFuturePrediction
!pip install -r requirements.txt
```

If your project is already in Google Drive, use:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/BinanceTrading
!ls
```

## 2. Optional: persist data and models
### Google Colab with Google Drive
```python
import os
os.environ['BFP_STORAGE_DIR'] = '/content/drive/MyDrive/BinanceFuturePredictionRuntime'
```

### Cloud VM or local custom storage
```bash
export BFP_STORAGE_DIR=/opt/binance_future_prediction_runtime
```

If `BFP_STORAGE_DIR` is not set, runtime data is stored in the project folder.

## 3. Select provider
### Notebook environment example for Bybit
```python
import os
os.environ['BFP_PROVIDER'] = 'bybit'
os.environ['BYBIT_MODE'] = 'testnet'
os.environ['BYBIT_API_KEY'] = 'YOUR_BYBIT_TESTNET_KEY'
os.environ['BYBIT_API_SECRET'] = 'YOUR_BYBIT_TESTNET_SECRET'
```

### Most reliable Colab commands
Pass the provider directly to the script so Colab does not depend on environment propagation between cells and shell commands.

```python
!python scripts/train_all.py --provider bybit --mode testnet --storage-dir /content/drive/MyDrive/BinanceFuturePredictionRuntime
!python scripts/backtest_all.py --provider bybit --mode testnet --storage-dir /content/drive/MyDrive/BinanceFuturePredictionRuntime
!python scripts/auto_trade_bot.py --provider bybit --mode testnet --storage-dir /content/drive/MyDrive/BinanceFuturePredictionRuntime
```

### Binance example
```python
import os
os.environ['BFP_PROVIDER'] = 'binance'
os.environ['BINANCE_MODE'] = 'testnet'
os.environ['BINANCE_API_KEY'] = 'YOUR_BINANCE_TESTNET_KEY'
os.environ['BINANCE_API_SECRET'] = 'YOUR_BINANCE_TESTNET_SECRET'
```

By default, training and downloading use `market_data_mode=production` even if trading mode is `testnet`, which keeps historical market data closer to real conditions.

Optional overrides:
```python
os.environ['BFP_DEFAULT_LEVERAGE'] = '10'
os.environ['BFP_FIXED_MARGIN_USDT'] = '10'
os.environ['BFP_USE_FULL_ACCOUNT_BALANCE'] = 'false'
os.environ['BFP_BALANCE_USAGE_FRACTION'] = '0.98'
os.environ['BYBIT_CATEGORY'] = 'linear'
os.environ['BYBIT_SETTLE_COIN'] = 'USDT'
os.environ['BFP_MARKET_DATA_MODE'] = 'production'
```

## 4. Run commands
If you already set environment variables in the notebook:

```python
!python scripts/train_all.py
!python scripts/backtest_all.py
!python scripts/auto_trade_bot.py
```

If Colab still picks the wrong provider, use the explicit `--provider` command form shown above.

## Notes
- Your existing local Binance JSON config still works.
- Legacy Binance-only settings are still accepted and automatically migrated in memory.
- Provider data is separated so Bybit models do not overwrite Binance models.
- Colab is fine for training and backtesting, but not ideal for long-running live bots because sessions can disconnect.
- A cloud VM is the better choice for always-on auto trading.
