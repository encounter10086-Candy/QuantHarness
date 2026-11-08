# LLM Trading Agent Backtest
This project implements an LLM-powered trading agent for BTC-USD 4-hour crypto market backtesting, replicating and modifying a multi-agent trading framework.

The original architecture consists of the Indicator Agent, Pattern Agent, Trend Agent and Decision Agent. We switch the underlying model to DeepSeek. Since DeepSeek cannot process chart images, the Indicator Agent and Pattern Agent are removed. We add confidence-based signal filtering as our modification for this assignment.

## Environment Setup
### 1. Create & activate virtual environment
```bash
# create virtual environment
python3 -m venv venv

# activate (Mac / Linux)
source venv/bin/activate

### 2. Install dependencies
```bash
pip install yfinance pandas numpy
```
> This project uses `yfinance` to download BTC-USD historical OHLC data.

## Configuration
- `ENABLE_CONFIDENCE_FILTER`: Toggle baseline / modified version
  - `False`: Baseline agent (original logic, no confidence filter)
  - `True`: Modified agent with confidence filtering (threshold = 0.6)
- `CONF_THRESHOLD = 0.6`: Minimum confidence required to open a LONG/SHORT position
- `TICKER = "BTC-USD"`, `INTERVAL = "4h"`, `PERIOD = "15d"`
- `REPEAT = 3`: Number of repeated backtest trials

## Run Backtest
```bash
python3 backtest.py
```

### Execution workflow
1. Download BTC 4-hour OHLC data from Yahoo Finance
2. Run rolling-window backtest for repeated trials
3. Print real-time decision, confidence, trade count and current equity
4. Output summary statistics including mean and standard deviation of metrics
5. Save raw experimental results

## Output Metrics
- `cumulative_return`: Total return relative to initial capital
- `win_rate`: Proportion of closed trades with positive PnL
- `max_drawdown`: Worst peak-to-valley loss
- `n_trades`: Number of executed trades

## Modification Description
We implement confidence-based signal filtering. The LLM Decision Agent outputs a confidence score ranging from 0 to 1 alongside each LONG/SHORT decision. A position will only be opened if the confidence score is at least 0.6; low-confidence signals are skipped.

## Notes
1. The Indicator Agent and Pattern Agent are removed. DeepSeek lacks image parsing capability and cannot interpret visual charts. Only the Trend Agent and Decision Agent are retained.
2. LLM outputs are stochastic. Slight differences may be observed across separate runs.
3. This backtest **does not implement the risk-constrained execution rules from the original paper**.
4. A valid DeepSeek API key must be configured in the `.env` file for the agent to run properly.
