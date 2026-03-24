# Run Anywhere Guide

This project now supports multiple exchange providers and can run locally, in Google Colab, or on cloud VMs like Oracle Cloud, AWS, Hetzner, or DigitalOcean.

## Supported providers
- Binance Futures
- Bybit Linear Futures

The runtime is selected with either:
- `local/trading_config.json`
- environment variables

Environment variables override file settings.

## 1. Install dependencies
```python
!git clone <your-repo-url>
%cd BinanceFuturePrediction
!pip install -r requirements.txt
```

## 2. Optional: persist data and models
### Google Colab with Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

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
### Binance example
```python
import os
os.environ['BFP_PROVIDER'] = 'binance'
os.environ['BINANCE_MODE'] = 'testnet'
os.environ['BINANCE_API_KEY'] = 'YOUR_BINANCE_TESTNET_KEY'
os.environ['BINANCE_API_SECRET'] = 'YOUR_BINANCE_TESTNET_SECRET'
```

### Bybit example
```python
import os
os.environ['BFP_PROVIDER'] = 'bybit'
os.environ['BYBIT_MODE'] = 'testnet'
os.environ['BYBIT_API_KEY'] = 'YOUR_BYBIT_TESTNET_KEY'
os.environ['BYBIT_API_SECRET'] = 'YOUR_BYBIT_TESTNET_SECRET'
```

By default, training/downloading uses `market_data_mode=production` even if trading mode is `testnet`, which keeps historical market data closer to real conditions.

Optional overrides:
```python
os.environ['BFP_DEFAULT_LEVERAGE'] = '10'
os.environ['BFP_FIXED_MARGIN_USDT'] = '10'
os.environ['BFP_USE_FULL_ACCOUNT_BALANCE'] = 'false'
os.environ['BFP_BALANCE_USAGE_FRACTION'] = '0.98'
```

Optional provider-specific overrides:
```python
os.environ['BYBIT_CATEGORY'] = 'linear'
os.environ['BYBIT_SETTLE_COIN'] = 'USDT'
os.environ['BFP_MARKET_DATA_MODE'] = 'production'
```

## 4. Run commands
```python
!python scripts/train_all.py
!python scripts/backtest_all.py
```

```python
!python scripts/auto_trade_bot.py
```

## Notes
- Your existing local Binance JSON config still works.
- Legacy Binance-only settings are still accepted and automatically migrated in memory.
- Provider data is separated so Bybit models do not overwrite Binance models.
- Colab is fine for training and backtesting, but not ideal for long-running live bots because sessions can disconnect.
- A cloud VM is the better choice for always-on auto trading.
