# Colab Run Guide

This project now supports both local and Google Colab runs without changing the current local setup.

## 1. Install dependencies
```python
!git clone <your-repo-url>
%cd BinanceFuturePrediction
!pip install -r requirements.txt
```

## 2. Optional: persist data/models to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os
os.environ['BFP_STORAGE_DIR'] = '/content/drive/MyDrive/BinanceFuturePredictionRuntime'
```

If `BFP_STORAGE_DIR` is not set, data/models are stored in the cloned repo folder.

## 3. Configure Binance credentials for testnet or production
You can keep using `local/trading_config.json` locally.

In Colab, the easiest way is environment variables:
```python
import os
os.environ['BINANCE_MODE'] = 'testnet'
os.environ['BINANCE_API_KEY'] = 'YOUR_TESTNET_KEY'
os.environ['BINANCE_API_SECRET'] = 'YOUR_TESTNET_SECRET'
```

Optional runtime overrides:
```python
os.environ['BFP_DEFAULT_LEVERAGE'] = '10'
os.environ['BFP_FIXED_MARGIN_USDT'] = '10'
os.environ['BFP_USE_FULL_ACCOUNT_BALANCE'] = 'false'
os.environ['BFP_BALANCE_USAGE_FRACTION'] = '0.98'
```

## 4. Run training / backtest / bot
```python
!python scripts/train_all.py
!python scripts/backtest_all.py
```

```python
!python scripts/auto_trade_bot.py
```

## Notes
- Local JSON config still works exactly as before.
- Environment variables override file settings when both are present.
- Training/backtesting can run without Binance API credentials if you already have cached data.
- Live trading on Colab works technically, but Colab is not ideal for always-on bot hosting because sessions can disconnect.
