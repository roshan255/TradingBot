# Crypto Futures ML Bot

A simple machine-learning crypto futures bot that can:
- download market data
- build features
- train a model per symbol
- backtest signals
- run an auto-trading loop

It supports multiple providers:
- Binance Futures
- Bybit Linear Futures

The bot can run locally, in Google Colab, or on cloud VMs.

## What the bot does

### Training
For each symbol, the bot:
1. downloads or refreshes candle data
2. builds technical features
3. trains multiple model candidates
4. selects the best model for that symbol
5. saves the model and metadata

### Backtesting
The bot loads saved models and simulates trades on the held-out test set.

### Auto trading
The bot:
1. scans all configured symbols
2. gets the latest candles
3. builds the latest feature row
4. predicts SHORT / NO TRADE / LONG probabilities
5. chooses the best trade
6. opens one position only
7. sets take profit and stop loss
8. monitors the open position and closes it if the signal flips

## Project structure

- `src/binance_future_prediction/`
  Core package code
- `scripts/train_all.py`
  Train all symbols
- `scripts/backtest_all.py`
  Backtest all symbols
- `scripts/auto_trade_bot.py`
  Run the live bot
- `local/trading_config.example.json`
  Example runtime config
- `Data/`
  Generated market data, features, models, and metadata

## Features used by the model

The model uses a mix of:
- trend features: SMA, EMA, MA gaps, MA slopes
- momentum features: RSI, ROC, MACD, stochastic, Williams %R
- volatility features: ATR, ATR%, Bollinger width, rolling volatility
- volume features: volume ratios, MFI, OBV, OBV changes
- candle structure: body, wick sizes, body ratio, range percent
- lag features: previous close, previous RSI, return history
- time features: hour of day, day of week

## Exchange support

The project now uses a provider abstraction.

Current providers:
- `binance`
- `bybit`

This means the same training and trading flow can run on different exchanges by changing config only.

Provider-specific data and models are kept separate so one provider does not overwrite another.

## Runtime config

Copy the example file if you want file-based config:

```powershell
Copy-Item local\trading_config.example.json local\trading_config.json
```

Main config fields:
- `provider`
  Selects the active exchange
- `providers.binance`
  Binance credentials and mode
- `providers.bybit`
  Bybit credentials and mode
- `trading.default_leverage`
  Default leverage
- `trading.fixed_margin_usdt`
  Fixed margin per trade
- `trading.use_full_account_balance`
  If `true`, uses nearly all available balance
- `trading.balance_usage_fraction`
  Safety fraction when full-balance mode is enabled

## Environment variable support

You can also run without a local config file.

Examples:

### Binance
```powershell
$env:BFP_PROVIDER='binance'
$env:BINANCE_MODE='testnet'
$env:BINANCE_API_KEY='YOUR_KEY'
$env:BINANCE_API_SECRET='YOUR_SECRET'
```

### Bybit
```powershell
$env:BFP_PROVIDER='bybit'
$env:BYBIT_MODE='testnet'
$env:BYBIT_API_KEY='YOUR_KEY'
$env:BYBIT_API_SECRET='YOUR_SECRET'
```

Optional portable storage path:
```powershell
$env:BFP_STORAGE_DIR='D:\BotRuntime'
```

## Install

```powershell
pip install -r requirements.txt
```

## Commands

### Train all symbols
```powershell
python scripts\train_all.py
```

### Backtest all symbols
```powershell
python scripts\backtest_all.py
```

### Run the auto-trading bot
```powershell
python scripts\auto_trade_bot.py
```

Optional runtime overrides:

```powershell
python scripts\train_all.py --provider bybit --mode testnet
python scripts\backtest_all.py --provider bybit --mode testnet
python scripts\auto_trade_bot.py --provider bybit --mode testnet
```

## Training output meaning

Important fields printed during training:
- `Accuracy`
  Overall test accuracy
- `Baseline Accuracy`
  Accuracy of always predicting the most common class
- `Macro F1`
  Average class quality across SHORT / NO TRADE / LONG
- `Balanced Accuracy`
  Average recall across the 3 classes
- `Trade Precision`
  Accuracy only on predicted LONG and SHORT trades
- `Signal Threshold`
  Minimum probability needed to allow a trade signal
- `Class Gap`
  Minimum probability gap between the top class and the next class

`Baseline Accuracy` matters a lot. If `Accuracy` is below `Baseline Accuracy`, the model is not beating a naive majority-class predictor.

## Storage behavior

The bot writes files safely using atomic replacements for:
- `data.csv`
- `features.csv`
- `model.pkl`
- `model_meta.json`

That means rerunning training should not leave half-written files behind if a run stops in the middle.

## Colab and cloud

The bot can run in:
- local Windows/Linux/macOS
- Google Colab
- Oracle Cloud / AWS / other VMs

For Colab and cloud setup, see `COLAB.md`.

Important Colab note:
- use `%cd /content/drive/MyDrive/...`
- do not use `!cd ...` if you want the directory change to persist

## Notes

- Colab is fine for training and backtesting.
- Colab is not ideal for long-running live trading because sessions can disconnect.
- A cloud VM is better for always-on trading.
- Existing local Binance-only config is still supported.

