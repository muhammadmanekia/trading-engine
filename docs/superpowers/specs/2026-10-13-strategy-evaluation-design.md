# Strategy evaluation harness (Part 0 of "find strategies that beat buy-and-hold")

Date: 2026-10-13. Status: design, awaiting review.

## Goal and context

Find stock strategies on Alpaca (paper) that beat buy-and-hold on data they were not tuned on. Robinhood is out of
scope for now. Everything is long-only and cash-funded: no margin, no shorting.

The overall effort splits into independent parts, each with its own design, plan and build:

| Part | What | Status |
|---|---|---|
| 0 | Evaluation harness (this document) | designing |
| 1 | Momentum rotation (hold the strongest N of the universe, rotate periodically) | later; needs cross-symbol ranking |
| 2 | Trend following with a ratcheting stop, re-entry and pyramiding | later; needs three engine features |
| 3 | News/AI strategy | already built; running propose-only on `alpaca-paper`, collecting forward evidence |
| 4 | Intraday strategies | last |

Parts 1, 2 and 4 are judged by this harness. Part 3 cannot be backtested (a model may remember history); it is
judged by `llm-report` on forward data and is outside this harness.

## Decisions already made

* **Success** = beat buy-and-hold on total return on unseen data (option A). Drawdown is reported, not gated.
* **Universe** = the 25 large caps in `strategies/examples/core-trend-hold.toml`. Widening to ~100 stocks is a
  later robustness rerun for the best candidates only.
* **Protocol** = rolling windows while building, then one untouched final exam.
* **Reality check on the benchmark:** the backtester's existing "buy & hold" figure has not been verified for a
  25-stock universe, so this harness computes its own.

## The yardstick

Equal dollar amounts of the 25 stocks bought at the first bar's open and held to the end, no rebalancing. It pays
the same spread and slippage as strategies, once, at purchase. Every strategy starts with the same cash.
Each strategy is compared with buy-and-hold of **its own** universe (identical to the shared 25 stocks today).

## What is reported

Per strategy and per window (and for the exam): total return, the benchmark's return, excess return over the
benchmark, number of trades, maximum drawdown, and the parameters chosen. Per-window Sharpe ratio and fraction of
time invested are not shown by `compare` (the full `run` command still reports them).

## "Beats buy-and-hold" (all four must hold)

1. **Final exam:** net return on the untouched test period (about mid-2024 to now) exceeds the benchmark's.
2. **Rolling windows:** excess return is positive in at least two thirds of the windows (with the ~4 windows that
   mid-2020 to mid-2024 allows, that means at least 3 of 4).
3. **Trade count:** at least 30 trades counted across all rolling windows and the final exam together, so the
   result is not one lucky position.
4. **Cost stress:** conditions 1 and 2 still hold with spread, slippage and fees doubled. This is a full rerun of
   the protocol under the doubled costs: parameters are re-chosen on each training window under those costs, so
   it can pick different parameters than the base run. It is not the base run's chosen parameters re-priced.

Passing all four makes a strategy a candidate for a paper run. Passing some is reported as "not proven".
A drawdown worse than 40% from peak in any window is flagged (visible, not a pass/fail condition).

## Components

* `engine/backtest/benchmark.py`: the equal-weight buy-and-hold using the same cost model as the engine.
* `engine/backtest/compare.py`: builds the rolling windows and the exam period, runs each strategy through the
  existing `BacktestEngine` on each, applies the four conditions, and maintains the exam ledger.
* `compare` command in `engine.backtest` (CLI): `compare --strategy a.toml --strategy b.toml [--final-exam]`.
  Prints a scoreboard and, with `--out`, writes it as JSON and text (`scoreboard.json`, `scoreboard.txt`); no
  CSV. Per-window Sharpe ratio and time invested are not shown (the full `run` command still reports them).
  Reuses the cached Alpaca bars.

It uses the real backtest engine, so results keep matching what trades live. Speed is limited by design: each
strategy gets only a small grid of settings, which limits over-fitting.

## Data flow

Cached daily bars for the 25 stocks → rolling windows (tune on 2 years, score on the next 6 months, then step forward 6 months;
windows never overlap with what they are scored on) → each strategy on each window, plus a doubled-cost rerun →
scoreboard. The final exam runs only with `--final-exam`, after the rolling stage.

## Keeping the exam honest

The final exam result is written to `runs/exam-ledger.json` (under the project folder, wherever the command is
run from), keyed by the strategy id, the strategy file's hash and a fingerprint of the settings that shape the
result (grid, objective, window sizes, exam start, end date, risk config, venue costs, cash, volume scale). Only
the first exam of a strategy id is clean. If a strategy is edited after its exam result was recorded, or the same
file is examined again under different settings, the scoreboard labels it "exam already seen by an earlier
version" and does not count it as a clean pass. Rerunning the identical file with identical settings stays clean.
This prevents tuning until the exam looks good. Exams are recorded only after every strategy in the run has been
evaluated, so a run that fails partway uses up nothing.

What the ledger does not stop:

* renaming the strategy `id`, which starts a fresh, "clean" history;
* submitting many differently-named variants in one run and keeping the best one;
* deleting the ledger file or pointing `--ledger` somewhere else (a new ledger prints a note, nothing more);
* running from another directory (fixed: the default ledger is always the project's `runs/exam-ledger.json`);
* the ledger is a local file (`runs/` is git-ignored), not shared and not tamper-proof.

It is a guard against fooling yourself, not against a determined cheater.

## Error handling

The run stops with a clear message, and nothing is skipped silently, when: a stock has no data, a window has too
few bars, a strategy fails to load, or the cached data has gaps the existing data-quality checks reject.

## Testing

* Benchmark math is exact on small synthetic price series, including costs.
* Training and scoring windows never overlap, and every bar falls in at most one scoring window.
* The cost-stress run applies exactly double spread and slippage.
* The exam ledger flags a strategy edited after its exam.
* A never-trading strategy has excess equal to minus the benchmark; a hold strategy is positive but below the
  benchmark (position caps keep it under 100% invested).
* No bar at or after a window's scored period (or after the exam start, for rolling windows) can change that
  window's parameters or score (perturbation tests); the strategy's first possible fill lines up with the
  benchmark's first open.
* The four-part rule is unit-tested at its boundaries (one window short of two thirds, 29 vs 30 trades).

## Known limits

* Survivorship and hindsight bias: the 25 stocks are today's large caps, chosen with hindsight, so trend and
  momentum strategies are flattered relative to buy-and-hold. Any wider rerun should use point-in-time index
  membership, not today's list.
* About 6 years of daily history exist (from mid-2020), and the exam period is mostly a rising market.
  A strategy that handles crashes poorly could still pass the exam; the rolling windows (which include 2022)
  are the check on that.
* Free IEX data understates volume, so liquidity limits are approximate (existing `volume_scale` setting).
* Passing the harness is evidence, not proof. Candidates still need a paper run before any real money.

* The engine's older built-in benchmark covers only the first symbol; `compare` does not use it.
* The cost-stress rerun doubles the benchmark's entry cost as well as the strategy's, and re-tunes parameters
  under the doubled costs (a full rerun of the protocol, not the same chosen parameters).
* A rolling window where no parameter set qualified (only possible with a grid) holds cash and is flagged.

## Out of scope

Margin, shorting, Robinhood, intraday data, any change to how live orders are placed.
