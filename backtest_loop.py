# backtest_loop.py
import yfinance as yf
import pandas as pd
import numpy as np
from trading_graph import TradingGraph
from default_config import DEFAULT_CONFIG

def calculate_metrics(trade_list, initial_capital=10000):
    if len(trade_list) == 0:
        return {"cumulative_return": 0, "win_rate": 0, "max_drawdown": 0}
    capital = initial_capital
    capital_history = [capital]
    win_count = 0
    for trade in trade_list:
        pct_pnl = trade["pnl_pct"]
        capital = capital * (1 + pct_pnl)
        capital_history.append(capital)
        if pct_pnl > 0:
            win_count += 1
    cumulative_return = (capital - initial_capital) / initial_capital
    win_rate = win_count / len(trade_list)
    peak = np.maximum.accumulate(capital_history)
    drawdown = (capital_history - peak) / peak
    max_drawdown = np.min(drawdown)
    return {
        "cumulative_return": round(cumulative_return, 4),
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_drawdown, 4)
    }

def run_single_backtest(window_size=20, ticker="BTC-USD", period="30d", interval="4h"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df[["Open", "High", "Low", "Close"]].copy()
    graph = TradingGraph(DEFAULT_CONFIG)
    position = None
    entry_price = None
    trade_records = []
    for i in range(window_size, len(df)):
        window_data = df.iloc[i-window_size:i]
        res = graph.run(window_data)
        decision = res["decision"]
        current_price = float(df.iloc[i]["Close"])
        if position is None:
            if decision == "LONG":
                position = "LONG"
                entry_price = current_price
            elif decision == "SHORT":
                position = "SHORT"
                entry_price = current_price
        else:
            if (position == "LONG" and decision == "SHORT") or (position == "SHORT" and decision == "LONG"):
                if position == "LONG":
                    pnl = (current_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - current_price) / entry_price
                trade_records.append({"pnl_pct": pnl})
                position = None
                entry_price = None
    metrics = calculate_metrics(trade_records)
    return metrics

if __name__ == "__main__":
    repeat_times = 5
    all_results = []
    for run_id in range(repeat_times):
        print(f"===== Backtest run {run_id+1}/{repeat_times} =====")
        metric = run_single_backtest()
        print(metric)
        all_results.append(metric)
    print("\n===== All 5 runs result =====")
    for item in all_results:
        print(item)
