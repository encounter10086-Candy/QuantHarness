# backtest_loop.py
import yfinance as yf
import pandas as pd
import numpy as np
import json
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

def run_single_backtest(window_size=50, ticker="BTC-USD", period="3mo", interval="4h"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={"Date": "Datetime"})

    graph = TradingGraph(DEFAULT_CONFIG)
    position = None
    entry_price = None
    trade_records = []

    for i in range(window_size, len(df)):
        window_df = df.iloc[i-window_size:i]
        kline_dict = {
            "Datetime": window_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "Open": window_df["Open"].tolist(),
            "High": window_df["High"].tolist(),
            "Low": window_df["Low"].tolist(),
            "Close": window_df["Close"].tolist(),
            "Volume": window_df["Volume"].tolist(),
        }
        initial_state = {
            "kline_data": kline_dict,
            "analysis_results": None,
            "messages": [],
            "time_frame": "4hour",
            "stock_name": "BTC",
        }
        final_state = graph.graph.invoke(initial_state)
        res_str = final_state.get("final_trade_decision")
        res = json.loads(res_str)
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
