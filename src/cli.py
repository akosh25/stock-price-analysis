import argparse
from pathlib import Path
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True)      
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end",   default="2025-08-31")
    p.add_argument("--outfmt", choices=["csv", "parquet"], default="parquet")
    p.add_argument("--save-plots", action="store_true")
    return p.parse_args()

def download_prices(tickers, start, end) -> pd.DataFrame:
    """Visszaad: MultiIndex (Date, Ticker) indexű DataFrame oszlopokkal: Close"""
    data = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False
    )

    close = data["Close"]
    if isinstance(close, pd.Series):
        t = tickers[0] if isinstance(tickers, (list, tuple)) else tickers
        df = close.to_frame("Close").reset_index()
        df["Ticker"] = t
        long = df.set_index(["Date", "Ticker"]).sort_index()
    else:
        long = (close.reset_index()
                .melt(id_vars="Date", var_name="Ticker", value_name="Close")
                .dropna()
                .set_index(["Date", "Ticker"])
                .sort_index())
    return long

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """MA30, napi hozam, kum. hozam, normalizált ár (első nap=100) tickerenként."""
    def per_ticker(g):
        g = g.sort_index()
        g["MA30"] = g["Close"].rolling(30, min_periods=1).mean()
        g["Return"] = g["Close"].pct_change()
        g["CumReturn"] = (1 + g["Return"]).cumprod() - 1
        g["NormPrice"] = g["Close"] / g["Close"].iloc[0] * 100.0
        return g
    return df.groupby(level="Ticker", group_keys=False).apply(per_ticker)

def to_parquet(df: pd.DataFrame, base_dir: str):
    base = Path(base_dir)
    for t, g in df.groupby(level="Ticker"):
        out = base / f"ticker={t}" / "prices.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        g.to_parquet(out)

def save_price_plot(g: pd.DataFrame, ticker: str, outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(g.index, g["Close"], label="Close")
    plt.plot(g.index, g["MA30"], label="MA30")
    plt.title(f"{ticker} – Close vs MA30")
    plt.xlabel("Date"); plt.ylabel("USD"); plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{ticker}_close_ma30.png", dpi=150)
    plt.close()

def main():
    args = parse_args()
    df = download_prices(args.tickers, args.start, args.end)
    df = add_indicators(df)

    Path("data/curated").mkdir(parents=True, exist_ok=True)
    if args.outfmt == "parquet":
        to_parquet(df, base_dir="data/curated")
    else:
        df.to_csv("data/curated/prices.csv")

    if args.save_plots:
        for t in args.tickers:
            g = df.xs(t, level="Ticker")
            save_price_plot(g, t, outdir="reports")

if __name__ == "__main__":
    main()
