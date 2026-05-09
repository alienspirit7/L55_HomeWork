# Dueling DQN Stock Trading — Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` to execute task-by-task. Steps use `- [ ]` checkboxes. Each task is owned by a specific subagent role from the team (`data-analyst`, `data-scientist`, `frontend-developer`, `qa-engineer`, `product-manager`, `architect`). Track 3 workflow per project `CLAUDE.md`.

**Goal:** End-to-end Dueling Double DQN trading system on daily OHLCV data, with rate-limited Yahoo Finance pipeline, leak-free feature engineering, PyQt6 GUI, and a reproducible 3-seed backtest report.

**Architecture:** Clean layered Python 3.10+ codebase. `src/data/` owns the gatekeeper + features + splits. `src/env/` owns the RL trading environment. `src/models/` owns the Dueling-DQN network and Double-DQN learner. `src/training/` owns the training loop, replay buffer, ε-schedule, TensorBoard. `src/evaluation/` owns metrics and backtest. `src/gui/` owns the PyQt6 dashboard. All hyperparameters in `config/*.yaml`. Seeds threaded through `src/utils/seeding.py`.

**Tech Stack:** Python **3.12** (locked, not 3.14 — pandas_ta wheels not yet reliable on 3.14), PyTorch with **MPS** backend (Apple Silicon M4 Pro, 24 GB unified memory, 12 cores), gymnasium, yfinance, pandas, pandas_ta, numpy, PyQt6, pyqtgraph, mplfinance, tensorboard, pyyaml, pytest. venv per project. **Device priority:** `cuda → mps → cpu` (cuda kept for portability, mps used locally).

---

## Locked Decisions (from PRD + feedback + user input)

| Decision | Value | Source |
|---|---|---|
| Reward | Δ unrealized PnL of mark-to-market portfolio, minus transaction cost on Buy/Sell ticks | User |
| Action sizing | All-in / All-out | User |
| DQN extras | Double DQN on top of Dueling | User |
| Transaction cost | 10 bps of notional, applied on Buy and Sell | User |
| Evaluation seeds | 3 seeds, report mean ± std | User |
| Tracking | TensorBoard local logs | User |
| Sample data | input/META.csv, input/GOOG.csv, input/NVDA.csv (read-only) | User |
| Cache key | parquet by `{ticker}.parquet` storing full range, sliced in memory; TTL 24h before refetch | feedback |
| Rate limiter | token bucket: 10/min, 100/hr, max 2 concurrent, burst 5 per 10s | feedback |
| Normalization | fit on train split only, applied to val/test (rolling z-score for volume over 60d within train) | feedback |
| VWAP | documented as (H+L+C)/3 weighted by volume — daily approximation, called out in README | feedback |
| GUI confidence | argmax-Q margin (Q[a*] − second-best Q) plus softmax-Q gauge labelled "soft confidence" | feedback |
| Predict Next | action for the next trading day given features through latest available close; UI shows asof timestamp | feedback |
| Walk-forward | out-of-scope v1, noted as future work in README | feedback |
| Out-of-scope | live trading, broker integration, options/futures, multi-asset portfolios, intraday | feedback |
| Sharpe annualization | √252, costs included | feedback |
| Failure modes | NaN-loss guard rolls back to last ckpt; partial yfinance triggers Tier 3; history < window+horizon → user error; CUDA/MPS OOM halves batch and retries once | feedback |
| Python version | 3.12 locked (not 3.14 — pandas_ta wheels) | local check |
| Compute | Apple Silicon M4 Pro, 24 GB, MPS backend; cuda kept as code path for portability | local check |
| Disclaimer | README header + GUI footer: "educational project, not investment advice; yfinance has known data quality issues" | feedback |

---

## File Structure

```
L55_HomeWork/
├── README.md                          # PM owns; final deliverable
├── PRD_Dueling_DQN_Stock_Trading.md   # existing
├── PLAN.md                            # this file
├── requirements.txt                   # pinned
├── config/
│   ├── default.yaml                   # all hyperparameters
│   └── tickers.yaml                   # default tickers + date ranges
├── input/                             # READ-ONLY, sample CSVs
│   ├── META.csv
│   ├── GOOG.csv
│   └── NVDA.csv
├── output/
│   ├── models/                        # ckpt_{ticker}_seed{n}.pt
│   ├── runs/                          # TensorBoard logs
│   ├── backtests/                     # per-run metrics JSON + equity curves
│   └── analysis/                      # aggregated mean±std reports
├── data/
│   └── raw/                           # parquet cache (gitignored)
├── screenshots/                       # README assets
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── seeding.py                 # seed_everything(seed)
│   │   ├── device.py                  # detect cuda/cpu
│   │   └── config.py                  # YAML loader + dataclass
│   ├── data/
│   │   ├── gatekeeper.py              # ApiGatekeeper, token bucket
│   │   ├── fetcher.py                 # 3-tier fetch (parquet → yfinance → CSV)
│   │   ├── features.py                # 10-feature engineering, leak-free fit
│   │   └── splits.py                  # 70/15/15 temporal split
│   ├── env/
│   │   └── trading_env.py             # gymnasium-style env, all-in/out
│   ├── models/
│   │   └── dueling_dqn.py             # shared trunk + V/A heads + aggregator
│   ├── training/
│   │   ├── replay_buffer.py           # uniform replay
│   │   ├── trainer.py                 # Double DQN loop, ε schedule, TB
│   │   └── runner.py                  # multi-seed orchestration
│   ├── evaluation/
│   │   ├── backtest.py                # rolls out greedy policy on test split
│   │   ├── metrics.py                 # Total Return, Sharpe, MaxDD, WinRate, Sortino, Calmar, turnover, avg hold
│   │   └── benchmark.py               # buy-and-hold w/ same costs
│   └── gui/
│       ├── app.py                     # main window
│       ├── candlestick.py             # left panel chart
│       ├── analytics.py               # right panel gauge + telemetry + reasoning
│       └── workers.py                 # QThread workers for fetch/train/backtest
├── scripts/
│   ├── prepare_data.py                # CLI: fetch + cache + features + split
│   ├── train.py                       # CLI: train one seed
│   ├── backtest.py                    # CLI: run greedy backtest
│   ├── run_experiment.py              # CLI: 3 seeds + aggregate
│   ├── download_samples.py            # one-off: populate input/ CSVs
│   └── run_gui.py                     # launch PyQt6 app
└── tests/
    ├── test_gatekeeper.py
    ├── test_features_no_leak.py
    ├── test_env_step.py
    ├── test_dueling_aggregation.py
    ├── test_metrics.py
    └── test_backtest_smoke.py
```

---

## Phases & Tasks

Each task lists owning subagent, files, acceptance criteria. Each task ends with commit. Reviewer (separate fresh subagent) checks against this plan and project `CLAUDE.md` before next task starts.

---

### Phase 0 — Scaffolding

#### Task 0.1: Repository scaffold + venv + dependencies
**Owner:** `architect` (one-shot, then disband)
**Files:** create directory tree above; `requirements.txt`; `.gitignore` (venv, data/raw, output, __pycache__, *.pyc); `config/default.yaml` with full hyperparameter block; `config/tickers.yaml`.

- [ ] Create directory tree exactly as in File Structure section
- [ ] Write `requirements.txt` with pinned versions known-good on Python 3.12 + macOS arm64: `torch>=2.3,<2.6` (MPS supported), `gymnasium==0.29.*`, `yfinance==0.2.*`, `pandas==2.2.*`, `pandas_ta==0.3.*`, `numpy<2`, `PyQt6==6.7.*`, `pyqtgraph==0.13.*`, `mplfinance==0.12.*`, `tensorboard==2.16.*`, `pyyaml==6.*`, `psutil==5.9.*`, `pyarrow==15.*`, `pytest==8.*`
- [ ] Write `config/default.yaml` containing every hyperparameter (see Decisions table + DQN block: `gamma=0.99, lr=1e-4, batch=64, buffer=100000, target_sync_steps=1000, eps_start=1.0, eps_end=0.05, eps_decay_steps=50000, huber_delta=1.0, train_steps=200000, eval_every=5000, window=30, fee_bps=10, init_cash=10000`)
- [ ] Verify with: `python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python -c "import torch, yfinance, pandas_ta, PyQt6; print('mps:', torch.backends.mps.is_available())"`
- [ ] Commit: `chore: project scaffold and pinned deps`

#### Task 0.2: Seeding + device + config utilities
**Owner:** `data-scientist`
**Files:** `src/utils/seeding.py`, `src/utils/device.py`, `src/utils/config.py`, `tests/test_utils.py`

- [ ] TDD: write `tests/test_utils.py` covering `seed_everything(42)` reproducibility (two `torch.randn(3)` calls match), `pick_device()` returns valid torch.device, `load_config(path)` parses YAML into a frozen dataclass
- [ ] Implement `seed_everything`: seeds `random`, `numpy`, `torch`, `torch.cuda`, sets `torch.backends.cudnn.deterministic=True, benchmark=False`
- [ ] Implement `pick_device()` with priority `cuda → mps → cpu`: `torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`
- [ ] Implement `load_config` with PyYAML → `@dataclass(frozen=True)` Config
- [ ] Run `pytest tests/test_utils.py -v`; expect all green
- [ ] Commit: `feat(utils): seeding, device detection, config loader`

---

### Phase 1 — Data Pipeline

#### Task 1.1: ApiGatekeeper (token bucket rate limiter)
**Owner:** `data-analyst`
**Files:** `src/data/gatekeeper.py`, `tests/test_gatekeeper.py`

- [ ] TDD: tests for token bucket — fills at 10/min, hard cap 100/hr, max 2 concurrent (semaphore), burst of 5 in 10s allowed; sanitize_ticker rejects path traversal (`..`, `/`)
- [ ] Implement `ApiGatekeeper` with two token buckets (per-minute, per-hour), `threading.Semaphore(2)`, `acquire()` blocks or raises `RateLimitExceeded`
- [ ] Implement `sanitize_ticker(s)` returning `[A-Z0-9.\-]{1,10}` or raising
- [ ] All tests green
- [ ] Commit: `feat(data): ApiGatekeeper token-bucket rate limiter`

#### Task 1.2: 3-tier fetcher with parquet cache
**Owner:** `data-analyst`
**Files:** `src/data/fetcher.py`, `tests/test_fetcher.py` (mocks yfinance)

- [ ] TDD: tests cover parquet hit (Tier 1), parquet miss + yfinance success writes parquet (Tier 2), yfinance failure falls back to `input/{ticker}.csv` (Tier 3), TTL: file older than 24h triggers refetch, cache key is ticker-only and slicing returns requested date range
- [ ] Implement `fetch(ticker, start, end, gatekeeper)` returning a DataFrame with OHLCV columns
- [ ] Parquet path: `data/raw/{ticker}.parquet`; load full, slice by date
- [ ] Tier 3 path: `input/{ticker}.csv`; same slicing
- [ ] Use `gatekeeper.acquire()` before any yfinance call
- [ ] All tests green
- [ ] Commit: `feat(data): 3-tier fetcher with parquet cache and TTL`

#### Task 1.3: Sample CSV downloader
**Owner:** `data-analyst`
**Files:** `scripts/download_samples.py`

- [ ] Script downloads META, GOOG, NVDA daily bars 2018-01-01 → 2024-12-31 via yfinance (rate-limited via gatekeeper) and writes to `input/{TICKER}.csv`
- [ ] Manually run once; verify 3 CSVs exist and have ≥ 1500 rows each
- [ ] Note: input/ is read-only thereafter (per project CLAUDE.md)
- [ ] Commit: `data: sample CSVs for offline reproducibility`

#### Task 1.4: Feature engineering, leak-free
**Owner:** `data-analyst`
**Files:** `src/data/features.py`, `src/data/splits.py`, `tests/test_features_no_leak.py`

- [ ] TDD: leak test — fit normalizer on train slice only; verify val/test rows produced are byte-identical regardless of whether val/test are present at fit time (proves no leak)
- [ ] Implement 8 market features: `log_return`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `bbp`, `vwap_dist` (with VWAP = (H+L+C)/3 weighted by volume — documented), `volume_norm` (rolling 60d z-score, params from train only)
- [ ] Implement 2 agent features as zero placeholders (env injects them at step time)
- [ ] Implement `temporal_split(df, 0.7, 0.15, 0.15)` returning train/val/test slices in chronological order
- [ ] Implement `Normalizer.fit(train_df)` / `.transform(df)` for any features that need scaling (volume) — store mean/std from train
- [ ] All tests green
- [ ] Commit: `feat(data): leak-free feature engineering and temporal split`

#### Task 1.5: prepare_data.py CLI
**Owner:** `data-analyst`
**Files:** `scripts/prepare_data.py`

- [ ] CLI: `python scripts/prepare_data.py --ticker NVDA --start 2018-01-01 --end 2024-12-31 --config config/default.yaml`; outputs `output/processed/{ticker}.npz` with arrays + split indices + normalizer params
- [ ] Handles "history < window+horizon" with clear error message
- [ ] Commit: `feat(scripts): prepare_data CLI`

---

### Phase 2 — RL Environment

#### Task 2.1: Trading environment (gymnasium API)
**Owner:** `data-scientist`
**Files:** `src/env/trading_env.py`, `tests/test_env_step.py`

- [ ] TDD: tests cover `reset()` returns obs of shape `(window=30, n_features=10)`; `step(action)` returns `(obs, reward, terminated, truncated, info)`; action space `Discrete(3)`; Buy while flat opens position at next bar open with 10bps fee; Buy while long is no-op; Sell while flat is no-op; Sell while long closes at next bar open with 10bps fee; reward = Δ unrealized PnL minus any fees this tick; terminal at end of split
- [ ] Use next-bar-open execution to avoid look-ahead within the same bar
- [ ] Inject 2 agent features into obs each step: `position_flag` ∈ {0,1}, `unrealized_pnl_pct`
- [ ] All tests green
- [ ] Commit: `feat(env): all-in/all-out trading env with next-bar-open exec and 10bps fees`

---

### Phase 3 — Dueling Double DQN

#### Task 3.1: Dueling DQN network
**Owner:** `data-scientist`
**Files:** `src/models/dueling_dqn.py`, `tests/test_dueling_aggregation.py`

- [ ] TDD: forward returns shape `(batch, 3)`; aggregation matches `V + (A − A.mean(dim=-1, keepdim=True))`; gradient flows through both heads
- [ ] Architecture: shared trunk = flatten(window×features) → Linear(256) → ReLU → Linear(256) → ReLU; V head = Linear(256→128) → ReLU → Linear(128→1); A head = Linear(256→128) → ReLU → Linear(128→3)
- [ ] All tests green
- [ ] Commit: `feat(models): Dueling DQN with V/A streams and mean-centered aggregation`

#### Task 3.2: Replay buffer
**Owner:** `data-scientist`
**Files:** `src/training/replay_buffer.py`, `tests/test_replay.py`

- [ ] TDD: capacity FIFO eviction; `sample(batch)` returns tensors of correct shapes/dtypes; works on CPU and CUDA
- [ ] Implement uniform `ReplayBuffer(capacity)` with `(s, a, r, s', done)` storage as numpy arrays + `sample(batch_size)` returning torch tensors on target device
- [ ] All tests green
- [ ] Commit: `feat(training): uniform replay buffer`

#### Task 3.3: Double DQN trainer
**Owner:** `data-scientist`
**Files:** `src/training/trainer.py`, `tests/test_trainer_smoke.py`

- [ ] TDD smoke: 1k steps on synthetic env; loss decreases on average over the run; no NaNs
- [ ] Implement Double DQN target: `y = r + γ * Q_target(s', argmax_a Q_online(s', a))` (key Double-DQN line) with `Q_target` synced from `Q_online` every `target_sync_steps`
- [ ] Linear ε schedule from `eps_start` to `eps_end` over `eps_decay_steps`
- [ ] Huber loss (δ=1.0); Adam optimizer; gradient clip 10.0
- [ ] NaN guard: if loss is NaN, restore last good ckpt and halve LR; abort after 3 NaN events
- [ ] CUDA OOM guard: catch, halve batch size once, retry
- [ ] TensorBoard scalars: `loss`, `mean_q`, `epsilon`, `episode_return`, `eval/equity`
- [ ] Save ckpt every `eval_every` steps to `output/models/{ticker}_seed{n}_step{N}.pt`
- [ ] Smoke test green
- [ ] Commit: `feat(training): Double DQN trainer with TB, NaN guard, OOM retry`

#### Task 3.4: Multi-seed runner
**Owner:** `data-scientist`
**Files:** `src/training/runner.py`, `scripts/train.py`, `scripts/run_experiment.py`

- [ ] `train.py --ticker NVDA --seed 0 --config config/default.yaml`
- [ ] `run_experiment.py --ticker NVDA --config config/default.yaml` runs 3 seeds sequentially, then calls aggregator (Phase 4)
- [ ] Commit: `feat(training): single-seed and 3-seed runners`

---

### Phase 4 — Backtest, Metrics, Benchmark

#### Task 4.1: Metrics
**Owner:** `data-analyst`
**Files:** `src/evaluation/metrics.py`, `tests/test_metrics.py`

- [ ] TDD: known-input tests for Total Return, Sharpe (annualized √252, cost-inclusive), Max Drawdown, Win Rate, Sortino (downside std), Calmar (CAGR/MaxDD), turnover (sum of |Δposition| / mean equity), average holding period (bars)
- [ ] All tests green
- [ ] Commit: `feat(eval): metrics module with golden tests`

#### Task 4.2: Backtest + benchmark
**Owner:** `data-analyst`
**Files:** `src/evaluation/backtest.py`, `src/evaluation/benchmark.py`, `scripts/backtest.py`, `tests/test_backtest_smoke.py`

- [ ] `backtest(model, env)` rolls out greedy policy on test split; returns equity curve + trade log
- [ ] `buy_and_hold(env)` with identical starting capital and the same 10bps entry/exit fee
- [ ] `scripts/backtest.py --model output/models/NVDA_seed0_stepN.pt --ticker NVDA` writes `output/backtests/{ticker}_seed{n}.json` (metrics) and `equity.png`
- [ ] Smoke test: backtest a randomly-initialized model finishes without error
- [ ] Commit: `feat(eval): greedy backtest and buy-and-hold benchmark`

#### Task 4.3: Aggregator (mean ± std across seeds)
**Owner:** `data-analyst`
**Files:** extend `scripts/run_experiment.py`; `output/analysis/{ticker}_summary.md` template

- [ ] After 3 seeds, aggregate JSONs → write Markdown summary table with mean ± std for every metric, plus benchmark row
- [ ] Commit: `feat(eval): cross-seed aggregator with mean±std reporting`

---

### Phase 5 — GUI (PyQt6)

#### Task 5.1: App skeleton + threading
**Owner:** `frontend-developer`
**Files:** `src/gui/app.py`, `src/gui/workers.py`, `scripts/run_gui.py`

- [ ] Main window with top control bar (ticker text, two date pickers, 4 buttons: Prepare Data, Train Model, Run Backtest, Predict Next — sequentially enabled), status bar (device, models loaded, progress)
- [ ] `QThread`-based workers for fetch / train / backtest (do NOT block UI thread)
- [ ] Disclaimer footer: "Educational project — not investment advice. yfinance has known data quality issues."
- [ ] Manual smoke: launch GUI, click Prepare Data on NVDA → status shows progress
- [ ] Commit: `feat(gui): main window, control bar, async workers`

#### Task 5.2: Candlestick chart (left panel)
**Owner:** `frontend-developer`
**Files:** `src/gui/candlestick.py`

- [ ] Render OHLCV via pyqtgraph CandlestickItem (or mplfinance embed) with green close>open / red close<open
- [ ] Updates when Prepare Data finishes
- [ ] Commit: `feat(gui): candlestick chart panel`

#### Task 5.3: Analytics panel (right)
**Owner:** `frontend-developer`
**Files:** `src/gui/analytics.py`

- [ ] Action gauge: shows BUY/SELL/HOLD label + softmax-Q bar AND argmax-Q margin numeric (per locked decision)
- [ ] Telemetry: Memory % (psutil), CPU %, accelerator mem (`torch.cuda.memory_allocated` on cuda, `torch.mps.current_allocated_memory()` on mps, blank on cpu) — refreshed every 1s via QTimer; status bar shows active backend label ("CUDA" / "MPS" / "CPU")
- [ ] Reasoning text: heuristic mapping last-bar features → human-readable bullets ("RSI=23 → oversold", "MACD hist > 0 → bullish cross", etc.)
- [ ] Commit: `feat(gui): analytics, telemetry, reasoning panels`

#### Task 5.4: Wire Predict Next + Run Backtest end-to-end
**Owner:** `frontend-developer`
**Files:** modify `src/gui/app.py`

- [ ] Predict Next: loads latest ckpt for the ticker, runs forward on last `window` bars through latest close, displays action + asof timestamp
- [ ] Run Backtest: launches backtest worker, on completion shows metric table dialog and saves equity.png to `screenshots/`
- [ ] Manual smoke: run full happy path NVDA — Prepare → Train (100 steps for smoke) → Backtest → Predict Next; capture screenshots
- [ ] Commit: `feat(gui): wire predict-next and backtest dialogs`

---

### Phase 6 — QA & Documentation

#### Task 6.1: Reproducibility audit
**Owner:** `qa-engineer`

- [ ] Fresh clone in `/tmp` → `python -m venv venv && pip install -r requirements.txt` → `python scripts/run_experiment.py --ticker NVDA --config config/default.yaml`
- [ ] Verify Tier 3 (offline) path works: disconnect network, rerun → must succeed via input/NVDA.csv
- [ ] Verify two runs with same seed produce byte-identical metric JSONs
- [ ] Verify all tests pass: `pytest -v`
- [ ] If anything fails — file fix tasks back to owning agent in fresh sessions; never green-light unverified
- [ ] Commit: `chore: reproducibility audit notes`

#### Task 6.2: README — final deliverable
**Owner:** `product-manager`
**Files:** `README.md`

- [ ] Sections per project CLAUDE.md README Standard:
  1. Title + 1-sentence description
  2. Project schema diagram (mermaid or ASCII): data → env → DQN → backtest → GUI
  3. Data/process flow paragraph
  4. Setup: venv + `pip install -r requirements.txt`
  5. How to run: copy-pasteable commands for prepare_data → run_experiment → run_gui
  6. Results: screenshots/ with descriptive filenames (`gui_main.png`, `equity_curve_nvda.png`, `tensorboard_loss.png`, `action_gauge.png`); 3-seed mean±std table for META/GOOG/NVDA vs buy-and-hold
  7. Conclusions, observations, known limitations (VWAP approximation, yfinance quality, no walk-forward in v1, single asset)
  8. Disclaimer (educational, not advice)
  9. Out-of-scope list (live trading, brokers, options, multi-asset, intraday)
- [ ] Commit: `docs: README with results, screenshots, disclaimers`

---

## Self-Review Notes

- **Spec coverage:** every PRD section + every feedback bullet maps to a task — reward (2.1), DQN machinery (3.1-3.3), action sizing (2.1), costs (2.1, 4.2), leak (1.4), VWAP (1.4 + README), cache key (1.2), confidence (5.3), seeds/tracking/configs (0.1, 0.2, 3.4), evaluation depth (4.1, 4.3), disclaimer (5.1, 6.2), out-of-scope (6.2), failure modes (1.5, 3.3), walk-forward note (6.2), rate limiter algo (1.1), annualization (4.1), Predict Next semantics (5.4).
- **No placeholders:** every step states the actual command/code/file.
- **Type consistency:** `seed_everything`, `Normalizer.fit/transform`, `ApiGatekeeper.acquire`, `fetch`, `temporal_split`, `ReplayBuffer.sample`, `Trainer`, `backtest`, `buy_and_hold` used consistently across phases.

---

## Resolved Pre-Phase-0 Questions

1. **Repo:** Everything builds inside `L55_HomeWork/`. Standalone project, single git repo at this folder.
2. **Python:** 3.12 (locked). Local 3.14 too fresh for `pandas_ta` wheels.
3. **Compute:** Tuned for local M4 Pro + MPS. `train_steps=200000`, `batch=64`, `buffer=100000` — fits in 24 GB unified memory comfortably; full 3-seed run on one ticker should take ~30–60 min on MPS.
