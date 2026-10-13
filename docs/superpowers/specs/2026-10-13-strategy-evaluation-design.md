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

## What is reported

Per strategy and per window (and for the exam): total return, excess return over the benchmark, maximum
drawdown, Sharpe ratio, number of trades, and fraction of time invested.

## "Beats buy-and-hold" (all four must hold)

1. **Final exam:** net return on the untouched test period (about mid-2024 to now) exceeds the benchmark's.
2. **Rolling windows:** excess return is positive in at least two thirds of the windows (with the ~4 windows that
   mid-2020 to mid-2024 allows, that means at least 3 of 4).
3. **Trade count:** at least 30 trades across all windows, so the result is not one lucky position.
4. **Cost stress:** conditions 1 and 2 still hold with spread and slippage doubled.

Passing all four makes a strategy a candidate for a paper run. Passing some is reported as "not proven".
A drawdown worse than 40% from peak in any window is flagged (visible, not a pass/fail condition).

## Components

* `engine/backtest/benchmark.py`: the equal-weight buy-and-hold using the same cost model as the engine.
* `engine/backtest/compare.py`: builds the rolling windows and the exam period, runs each strategy through the
  existing `BacktestEngine` on each, applies the four conditions, and maintains the exam ledger.
* `compare` command in `engine.backtest` (CLI): `compare --strategy a.toml --strategy b.toml [--final-exam]`.
  Prints a scoreboard and writes JSON and CSV under `runs/`. Reuses the cached Alpaca bars.

It uses the real backtest engine, so results keep matching what trades live. Speed is limited by design: each
strategy gets only a small grid of settings, which limits over-fitting.

## Data flow

Cached daily bars for the 25 stocks → rolling windows (tune on 2 years, score on the next 6 months, then step forward 6 months;
windows never overlap with what they are scored on) → each strategy on each window, plus a doubled-cost rerun →
scoreboard. The final exam runs only with `--final-exam`, after the rolling stage.

## Keeping the exam honest

The final exam result is written to `runs/exam-ledger.json` keyed by the strategy file's hash. If a strategy is
edited after its exam result was recorded, the scoreboard labels the new version "exam already seen by an earlier
version" and does not count it as a clean pass. This prevents tuning until the exam looks good.

## Error handling

The run stops with a clear message, and nothing is skipped silently, when: a stock has no data, a window has too
few bars, a strategy fails to load, or the cached data has gaps the existing data-quality checks reject.

## Testing

* Benchmark math is exact on small synthetic price series, including costs.
* Training and scoring windows never overlap, and every bar falls in at most one scoring window.
* The cost-stress run applies exactly double spread and slippage.
* The exam ledger flags a strategy edited after its exam.
* A buy-and-hold strategy expressed as a strategy file scores about zero excess return.
* The four-part rule is unit-tested at its boundaries (one window short of two thirds, 29 vs 30 trades).

## Known limits

* About 6 years of daily history exist (from mid-2020), and the exam period is mostly a rising market.
  A strategy that handles crashes poorly could still pass the exam; the rolling windows (which include 2022)
  are the check on that.
* Free IEX data understates volume, so liquidity limits are approximate (existing `volume_scale` setting).
* Passing the harness is evidence, not proof. Candidates still need a paper run before any real money.

## Out of scope

Margin, shorting, Robinhood, intraday data, any change to how live orders are placed.
