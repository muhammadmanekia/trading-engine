# Writing, testing and running strategies

A strategy is a small TOML file. The same file is what you backtest, what paper-trades, and what trades live.
There is no code in it, so it cannot do anything the schema does not allow.

## 1. The file

```toml
id = "ema-cross-btc"                 # lowercase letters, digits, - and _
universe = ["BTC-USD"]               # one asset class per strategy ("BTC-USD" is crypto, "AAPL" is a stock)
timeframe = "1h"                     # "1h" or "1d"

[indicators]                         # name them anything; refer to them below
fast = { type = "ema", period = 12 }
slow = { type = "ema", period = 48 }
atr  = { type = "atr", period = 14 }

[entry]                              # when to buy (while flat)
all = [ { cross_above = ["fast", "slow"] }, { gt = ["close", { mul = ["slow", 0.99] }] } ]

[exit]                               # optional: when to sell (while holding). Without it, only the stop exits.
cross_below = ["fast", "slow"]

[stop]                               # REQUIRED: every entry carries a stop-loss
type = "atr"                         # or "percent" with pct = 0.05
multiple = 2.5

[sizing]
type = "risk_fraction"               # or fixed_notional / fixed_quantity
risk_pct = 0.01                      # lose at most 1% of the strategy budget if the stop is hit
```

Optional keys: `cooldown_bars` (wait after an exit), `max_positions`, `lookback` (history window),
`[take_profit]` (stocks on Alpaca only; Robinhood cannot hold one).

### Indicators
`sma`, `ema`, `rsi` (Wilder), `atr`, `stdev`, `highest`, `lowest`, `roc`, `bb_upper`, `bb_lower`.
Most take `period` and an optional `source` (`open high low close volume hl2 hlc3 ohlc4`); `bb_*` take `k`.

### Conditions
`gt gte lt lte eq`, `cross_above`, `cross_below` (each takes `[left, right]`), combined with `all`, `any`, `not`.
Operands are numbers, series names (`close`, an indicator), `{ ref = "slow", lag = 3 }` (three bars ago), or
arithmetic `{ add|sub|mul|div = [a, b] }`. Expressions are bounded in depth and size. No scripting.

## 2. How a signal becomes a trade
1. A bar **closes**. The strategy sees only closed bars (never the forming one).
2. The rule fires -> an **Intent** with a stop price and a size.
3. The **risk engine** may approve, shrink or refuse it (position caps, spread limit, daily loss, ...).
4. The order is placed and **fills at the next bar's open** (in a backtest; live it fills as soon as the market
   is open, which is the same thing).
5. The stop is placed with it: a bracket on Alpaca, a separate stop order on Robinhood.

## 3. Backtest it

```bash
python -m engine.backtest run --strategy strategies/examples/ema-cross-btc.toml \
    --venue robinhood-crypto --data alpaca --start 2025-01-01 --out runs/ema-cross
python -m engine.backtest run --strategy my.toml --venue alpaca-stocks --data csv:./my-data   # your own CSVs
```
CSV files are `<dir>/<SYMBOL>_<timeframe>.csv` with `timestamp,open,high,low,close,volume`.
`--data alpaca` needs `ALPACA_PAPER_KEY_ID` / `ALPACA_PAPER_SECRET` in your environment and caches to `.cache/bars`.

**Costs are real and pessimistic.** Every fill pays half the spread plus slippage against you. Robinhood's spread
is the big one: run `python -m engine.service quotes ...` for the live number, let the service run for a while
(it records quotes), then add `--spread-from-db` to use the measured median. Until then the default is a
200 bps placeholder, and the risk engine will refuse entries whose spread exceeds `max_slippage_bps` exactly
as it does live (the report tells you when that is why nothing traded).

### Reading the report
* **Return vs buy & hold**: a strategy that does not beat holding the asset, net of costs, is not worth running.
* **Max drawdown**: the worst peak-to-trough loss. Decide if you could sit through it.
* **Trades**: fewer than ~30 means the statistics are noise. The report warns you.
* **Avg R**: average profit in units of the risk taken per trade. Positive and stable is what you want.
* **Costs**: fees + spread/slippage as a share of starting equity. If costs exceed the whole result, the
  strategy trades too often for this venue.

## 4. Do not fool yourself

```bash
# Choose parameters ONLY from the past, then judge them on the data that follows
python -m engine.backtest walk-forward --strategy my.toml --venue alpaca-stocks --data alpaca \
    --grid indicators.fast.period=8,12,16 --grid stop.multiple=2,3 --train-bars 2000 --test-bars 500
# Nudge each parameter +/-20%: a real edge sits on a plateau, an overfit one on a needle
python -m engine.backtest sensitivity --strategy my.toml --venue alpaca-stocks --data alpaca \
    --param indicators.fast.period --param stop.multiple
```
Trust a result when out-of-sample efficiency is near 1, the chosen parameters stay stable window to window,
nearby parameter values are also profitable, and there are plenty of trades. Searching more combinations makes
overfitting *more* likely, which is why the grid is capped.

## 5. Run it for real
Put it in the account config and restart the service:
```toml
[[accounts]]
id = "alpaca-paper"
broker = "alpaca"
rules = ["strategies/examples/rsi-dip-stocks.toml"]
```
The service checks the strategy against the account's venue at start-up (for example it refuses a take-profit
on Robinhood) and then evaluates it once per newly closed bar. Stocks are only acted on while the market is
open. Restarts are safe: intent ids come from the bar, so the same bar can never trade twice.

**Prove live matches the test.** After some paper trading:
```bash
python -m engine.backtest replay --account alpaca-paper --config config/local.toml \
    --venue alpaca-stocks --data alpaca --timeframe 1d --symbols AAPL,MSFT,NVDA
```
It re-simulates the intents the account really received and prints, per trade, the real vs simulated fill and
the price error in basis points. Small, stable errors mean the backtest can be trusted; large ones mean the cost
model needs fixing before you believe any backtest.

## 6. Beyond rules (news, LLMs)
A rule strategy is one kind of `SignalSource`: something that looks at the world once per bar and proposes
Intents. A news listener or an LLM strategy is another. It proposes trades; it never places them and never
holds broker keys. The same risk engine, order manager and reconciler judge and execute what it proposes, so a
wrong or manipulated news item can at worst propose a trade the risk engine then limits or refuses (position
caps, spread, daily loss, kill switch). Backtesting such a source needs an archive of historical news, which is
a data problem to solve when we build it.
