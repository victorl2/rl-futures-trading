import numpy as np
import pandas as pd
import time
from binance import Client


def download_candlesticks(symbol, start, end):
    """
    Download candlestick from Binance for the given symbol.
    """
    client = Client()
    klines = client.get_historical_klines(
        "BNBBTC", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
    return klines


class TradingEnv:
    def __init__(self):
        self.candlesticks = np.array()

    def step(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass


if __name__ == "__main__":
    candles = download_candlesticks("BNBBTC", "1 Jan, 2019", "1 Feb, 2019")
    for candle in candles:
        print(candle)
