# Product Requirements Document (PRD)

**Project:** Dueling DQN Stock Trading System

---

## 1. Project Overview & Objective

The objective of this project is to build an end-to-end Reinforcement Learning (RL) environment and agent for algorithmic trading. The system allows users to train an AI agent to trade financial assets (stocks) by interacting with historical market data.

The agent learns to maximize its return while minimizing risk using technical indicators as its state representation. The project includes a full data pipeline, a custom RL model, a rate-limiting Gatekeeper, and a Graphical User Interface (GUI).

**Critical Constraint:** The core algorithm must be a Dueling Deep Q-Network (Dueling DQN), not a standard DQN.

---

## 2. Core Algorithm: Dueling DQN

As emphasized in the lecture, the system must implement a Dueling DQN architecture. This architecture separates the estimation of the Q-value into two separate streams:

- **Value Stream $V(s)$ (Scalar):** Evaluates the "State Quality." It determines how good it is to be in the current market state, regardless of the action taken. (e.g., In a strong bull market, simply being in the market is good).
- **Advantage Stream $A(s, a)$ (Vector):** Evaluates the "Action Advantage." It calculates the relative benefit of taking a specific action (Buy, Sell, Hold) compared to the other possible actions in that specific state.
- **Aggregation:** The network combines these two streams at the output layer to calculate the final Q-values: $Q(s, a) = V(s) + A(s, a) - \text{mean}(A)$.

**Action Space:**

The agent can take three discrete actions:

- `0`: Hold
- `1`: Buy
- `2`: Sell

---

## 3. Data Architecture & Pipeline

The application handles live data fetching, local caching, and feature engineering.

### 3.1 Data Source & Base Metrics

- **Source:** Yahoo Finance via the Python `yfinance` library.
- **Asset Class:** Equity OHLCV bars (Open, High, Low, Close, Volume).
- **Granularity:** Daily data (`interval="1d"`).

### 3.2 The 3-Tier Fallback System & Gatekeeper

To prevent API rate-limiting blocks and provide offline resilience, data fetching must go through an `ApiGatekeeper` using a 3-tier strategy:

- **Tier 1 (Primary): Parquet Cache.** Looks for local compressed `.parquet` files (`data/raw/{ticker}_{start}_{end}.parquet`). If found, returns immediately.
- **Tier 2: Live API via Gatekeeper.** If not in cache, makes a live call to `yfinance`.
  - Rate Limits enforced by Gatekeeper: 10 requests per minute, 100 requests per hour (max 2 concurrent, burst 5/10s).
  - Post-fetch: Results are immediately written to the Parquet cache for future use.
- **Tier 3: Offline CSV Fallback.** If the live API fails (e.g., no internet, rate limit exceeded), the system looks for a local fallback CSV (`data/raw/{ticker}.csv`).

**Note on Gatekeeper Security:** As mentioned in the lecture, the gatekeeper also acts as a security abstraction layer, preventing direct unthrottled API access and sanitizing file naming/handling.

### 3.3 Feature Engineering (State Representation)

The raw OHLCV data is processed using the `pandas_ta` library to create a 10-feature state vector per timestep:

- **8 Market Indicators:** Log Return, RSI (14), MACD, MACD Signal, MACD Histogram, Bollinger Bands %B, VWAP Distance, Volume Normalization (min-max scaled).
- **2 Agent State Features:** Agent's current position (holding/not holding) and Unrealized PnL.
- **Observation Window:** The agent observes a sliding window of the last 30 days of these 10 features.
- **Data Split:** The dataset is split temporally into 70% Training / 15% Validation / 15% Testing.

---

## 4. Graphical User Interface (GUI) Requirements

The system must include an interactive dashboard (e.g., built with PyQt6) to visualize the data, the training process, and the agent's predictions.

### 4.1 Layout Overview

**Top Control Bar (Inputs):**

- Ticker Input: Text field for the stock symbol (e.g., AAPL, NVDA).
- Start/End Dates: Date pickers to define the range for data fetching.
- Action Buttons: Sequentially enabled buttons for: Prepare Data, Train Model, Run Backtest, Predict Next.

**Left Panel (Candlestick Chart):**

- Displays historical price data visually.
- Green candles indicate bullish days (Close > Open).
- Red candles indicate bearish days (Close < Open).

**Right Panel (Analytics & Telemetry):**

- Action Gauge: Displays the model's predicted action (BUY, SELL, HOLD) and confidence level as a progress bar.
- System Telemetry: Real-time hardware monitoring showing Memory, CPU, and GPU (CUDA) usage.
- Reasoning: A text-based explanation interpreting the technical indicators driving the current prediction (e.g., "RSI is Oversold, MACD shows Bullish cross").

**Status Bar (Bottom):**

- Shows current hardware device (CPU/CUDA backend), number of loaded models, and progress messages/loading bars for ongoing tasks (like training epochs).

---

## 5. Key Performance Metrics (Backtesting)

When the "Run Backtest" action is triggered, the system must output the following metrics to evaluate the agent's performance against unseen data:

- **Total Return (%):** The overall percentage profit/loss generated by the agent. (Desired: Outperform a simple "buy and hold" strategy).
- **Sharpe Ratio:** A measure of risk-adjusted return. (Target: > 1.0 is acceptable, > 2.0 is excellent).
- **Max Drawdown (%):** The largest single drop from peak to trough in the portfolio's value. (Desired: As low as possible).
- **Win Rate (%):** The percentage of trades that resulted in a profit.

---

## 6. Hardware & System Requirements

- **Hardware Acceleration:** The system should automatically detect and utilize a GPU (NVIDIA CUDA) if available to speed up PyTorch neural network training. If unavailable, it should fall back to the CPU gracefully. The UI must display which backend is currently in use.
- **Python Environment:** Python 3.12+ using `uv` or `pip` for dependency management (`torch`, `yfinance`, `pandas_ta`, `PyQt6`).

---

# Review Feedback

This is a solid PRD with good structure, but it has some significant gaps for an RL system specifically. The most important issues cluster around things that are essential for a working DQN but are completely unspecified.

## Critical Gaps

### Reward function is missing entirely

This is the single most important design decision in any RL project, and the document doesn't mention it. Options to specify:

- Per-step log return of portfolio
- Change in unrealized PnL
- Sharpe-based reward (differential Sharpe)
- Sparse end-of-episode return

Each produces dramatically different agent behavior. You also need to decide on penalties for transaction costs and possibly for excessive trading.

### Standard DQN machinery isn't specified

The doc covers the dueling architecture but skips the rest of what makes DQN work:

- Experience replay buffer (size, sampling strategy — uniform vs prioritized)
- Target network and its update frequency (hard sync every N steps, or Polyak averaging?)
- ε-greedy exploration schedule (start, end, decay)
- Whether to combine with Double DQN (very common pairing with Dueling)
- Discount factor γ, learning rate, batch size, optimizer, loss (Huber vs MSE)
- Network architecture for the shared trunk and the V/A heads

### Action semantics are ambiguous

"Buy" and "Sell" don't specify position size. Is it all-in / all-out? Fixed share count? % of equity? What happens when "Buy" is chosen while already long — no-op, or pyramid? These choices change the problem fundamentally.

### Transaction costs and slippage aren't mentioned

Without them, backtests will look spectacular and mean nothing. Even a flat 5–10 bps per trade will radically change which strategies look profitable.

## Technical Concerns

### Look-ahead bias risk in the feature pipeline

"Volume Normalization (min-max scaled)" — over what window? If it's over the full series, that's a leak. All normalization params must be fit on training data only and applied to val/test. Same goes for any feature scaling. Worth calling out explicitly.

### VWAP from daily bars is degenerate

True VWAP needs intraday data; on daily bars it's typically approximated as `(H+L+C)/3` weighted by volume, which provides much less signal than the name suggests. Either upgrade to intraday, swap the feature, or document what VWAP actually means here.

### Cache key is brittle

Keying parquet files on `{ticker}_{start}_{end}` means a request for AAPL 2020-01-01→2024-12-31 won't hit the cache for AAPL 2020-01-01→2024-12-30. Better to cache by ticker only, store the full available range, and slice in memory. Also no cache TTL/invalidation is specified — when does today's cached data go stale?

### "Confidence" from a DQN isn't well-defined

Q-values aren't probabilities. A softmax-over-Q gauge can work as a UI device, but it's worth noting this is a visualization choice, not a true confidence estimate. The argmax-Q margin is arguably more honest.

## Things to Add

### Reproducibility section

- Random seeds for numpy/torch/env
- Deterministic mode
- Experiment tracking (TensorBoard or Weights & Biases)
- Config files instead of hardcoded hyperparameters
- Saved model artifacts with the config that produced them

### More rigorous evaluation protocol

- Multiple seeds (≥5) per config with reported mean ± std — single-run RL results are nearly meaningless given training variance
- Add Sortino, Calmar, turnover, average holding period
- Clearly-marked benchmark (buy-and-hold on the same window with the same starting capital and costs)

### Educational/legal disclaimer

This trains an agent on real market data and outputs trade actions. Even if scoped as a learning project, the README and UI should make clear it's not investment advice and that yfinance data has known quality issues.

### Explicit out-of-scope list

No live trading, no broker integration, no options/futures, no portfolio of multiple assets simultaneously (or whatever the boundary is). PRDs are clearer about what they're building when they say what they're not.

### Failure modes

- What does the system do when training diverges (NaN losses)?
- When yfinance returns partial data?
- When the user picks a ticker with too little history for a 30-day window?
- When CUDA OOMs mid-training?

## Smaller Items

- The 70/15/15 split is fine, but for time series, walk-forward validation gives a more honest picture and is worth mentioning even if not implemented in v1.
- "Max 2 concurrent, burst 5/10s" in the rate limiter — specify the algorithm (token bucket? sliding window?) so the implementer doesn't have to guess.
- Mention disk space budget for the parquet cache and how old data is pruned.
- The Sharpe target ">1.0 acceptable, >2.0 excellent" should specify annualization assumption (√252 for daily) and whether costs are included.
- "Predict Next" in the GUI is underspecified — predict tomorrow's action given data through today's close? At what time? This matters for whether the system is implying it could be used in practice.

## Summary

Overall the architecture and scoping are reasonable; the biggest lift is filling in the RL-specific details (reward, replay, exploration, action sizing, costs) that determine whether the agent will actually learn anything useful.
