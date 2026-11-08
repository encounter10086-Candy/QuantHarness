"""
backtest loop
"""
import json
import re
import time
import numpy as np
import pandas as pd
import yfinance as yf

from trading_graph import TradingGraph
from default_config import DEFAULT_CONFIG

TICKER = "BTC-USD"
PERIOD = "15d"       
INTERVAL = "4h"
WINDOW_SIZE = 50      
STEP = 8              
REPEAT = 3             
INITIAL_CAPITAL = 10000.0
TIME_FRAME = "4hour"
STOCK_NAME = "BTC"

def fetch_ohlc():
    print(f"Fetching {TICKER} {INTERVAL} data for {PERIOD}...")
    df = yf.download(tickers=TICKER, period=PERIOD, interval=INTERVAL, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty data, aborting.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    if "Datetime" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "Datetime"})
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    print(f"Loaded {len(df)} candles from {df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]}")
    return df

def build_state(window_df):
    kline_dict = {
        "Datetime": window_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "Open": window_df["Open"].tolist(),
        "High": window_df["High"].tolist(),
        "Low": window_df["Low"].tolist(),
        "Close": window_df["Close"].tolist(),
        "Volume": window_df["Volume"].tolist(),
    }
    return {
        "kline_data": kline_dict,
        "analysis_results": None,
        "messages": [],
        "time_frame": TIME_FRAME,
        "stock_name": STOCK_NAME,
    }

def parse_decision(raw):
    if isinstance(raw, dict):
        return str(raw.get("decision", "HOLD")).upper()
    if not isinstance(raw, str):
        return "HOLD"
    text = raw.strip()
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return "HOLD"
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return "HOLD"
    return str(obj.get("decision", "HOLD")).upper()

def compute_metrics(trade_pnls, equity_curve, initial_capital=INITIAL_CAPITAL):
    if len(trade_pnls) == 0:
        return {"cumulative_return": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "n_trades": 0}
    cum_return = equity_curve[-1] / initial_capital - 1.0
    win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)
    peak = np.maximum.accumulate(np.array(equity_curve))
    dd = (np.array(equity_curve) - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    return {
        "cumulative_return": round(float(cum_return), 4),
        "win_rate": round(float(win_rate), 4),
        "max_drawdown": round(max_dd, 4),
        "n_trades": len(trade_pnls),
    }

def run_single_backtest(df, tg, run_id=1):
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trade_pnls = []
    position = None          
    entry_price = None
    start = WINDOW_SIZE
    stops = list(range(start, len(df), STEP))
    print(f"[run {run_id}] {len(stops)} decision points to evaluate...")

    for n, i in enumerate(stops, 1):
        window_df = df.iloc[i - WINDOW_SIZE:i]
        state = build_state(window_df)
        try:
            final_state = tg.graph.invoke(state)
            decision = parse_decision(final_state.get("final_trade_decision"))
        except Exception as e:
            print(f"[run {run_id}] step {n}/{len(stops)} failed: {e} -- skip")
            continue

        price = float(df.iloc[i]["Close"])

        if position is None:
            if decision == "LONG":
                position, entry_price = "LONG", price
            elif decision == "SHORT":
                position, entry_price = "SHORT", price
        else:
            exit_now = (
                (position == "LONG" and decision == "SHORT")
                or (position == "SHORT" and decision == "LONG")
            )
            if exit_now:
                pnl = (price - entry_price) / entry_price
                if position == "SHORT":
                    pnl = -pnl
                capital *= (1.0 + pnl)
                trade_pnls.append(pnl)
                equity_curve.append(capital)
                position, entry_price = None, None

        if n % 10 == 0 or n == len(stops):
            print(f"[run {run_id}] {n}/{len(stops)} done | decision={decision} "
                  f"| trades={len(trade_pnls)} | equity={capital:.2f}")
        time.sleep(0.3)  
        
    if position is not None:
        price = float(df.iloc[-1]["Close"])
        pnl = (price - entry_price) / entry_price
        if position == "SHORT":
            pnl = -pnl
        capital *= (1.0 + pnl)
        trade_pnls.append(pnl)
        equity_curve.append(capital)

    return compute_metrics(trade_pnls, equity_curve)

if __name__ == "__main__":
    df = fetch_ohlc()
    tg = TradingGraph(DEFAULT_CONFIG)

    all_results = []
    for r in range(1, REPEAT + 1):
        print(f"\n===== Backtest run {r}/{REPEAT} =====")
        metrics = run_single_backtest(df, tg, run_id=r)
        print(f"[run {r}] {metrics}")
        all_results.append(metrics)

    print("\n===== Summary over {} runs =====".format(REPEAT))
    for idx, m in enumerate(all_results, 1):
        print(f"run {idx}: {m}")

    keys = ["cumulative_return", "win_rate", "max_drawdown", "n_trades"]
    print("\n----- mean / std -----")
    for k in keys:
        vals = np.array([m[k] for m in all_results], dtype=float)
        print(f"{k}: mean={vals.mean():.4f}  std={vals.std():.4f}")

    with open("backtest_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved raw results to backtest_results.json")
