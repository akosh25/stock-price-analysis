import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2025-08-31")

data.to_csv("../data/aapl.csv")
data["MA30"] = data["Close"].rolling(window=30).mean()

#chart 
plt.figure(figsize=(12,6))
plt.plot(data["Close"], label="Closing Price")
plt.plot(data["MA30"], label="30-day Moving Average")
plt.title(f"{ticker} Stock Price (2020-2025)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
