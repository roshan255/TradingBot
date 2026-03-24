from _bootstrap import apply_runtime_cli


apply_runtime_cli()

from binance_future_prediction.live import generate_live_signals


def main() -> None:
    for signal in generate_live_signals():
        print("\nSymbol:", signal.symbol)
        print("Time:", signal.current_time)
        print("Current Price:", signal.price)
        print("Prediction Probability:", round(signal.probability, 3))
        print(signal.prediction)


if __name__ == "__main__":
    main()
