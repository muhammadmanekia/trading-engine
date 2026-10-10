# trading-engine

Event-sourced, fail-closed trading engine.

* **Phase 1:** event log, risk engine, order manager, mock broker.
* **Phase 2:** Alpaca (stocks) and Robinhood Crypto adapters, crypto paper simulator, execution service,
  reconciler, Discord alerts. See [docs/phase2-api-notes.md](docs/phase2-api-notes.md) for what was verified
  against the vendors' docs and what still needs the opt-in live smoke tests.

Design: [`../docs/superpowers/specs/2026-10-10-trading-bot-design.md`](../docs/superpowers/specs/2026-10-10-trading-bot-design.md)

```
Intent ──► Risk engine ──► Order manager ──► Broker (mock / paper / Alpaca / Robinhood)
              │                 │
              └──── append-only, hash-chained event log ◄──┘
                              │
                 AccountProjector (cash, positions, P&L, order states)
```

## Guarantees (each one is tested)

| Guarantee | How |
|---|---|
| Nothing reaches a broker except via `TradingCore.submit_intent` → `OrderManager` | `test_only_the_order_manager_talks_to_the_broker` |
| Risk engine is pure and never raises; any error or missing input is a rejection | `tests/property/test_risk_properties.py` |
| No approved entry can breach position, exposure, cash-reserve or budget limits | property test over generated accounts |
| Every entry has a stop-loss; an unattachable stop rejects the trade | `NO_STOP_LOSS`, `NO_PROTECTION` |
| Kill switch blocks entries immediately; exits and protective stops stay live | `test_core_e2e.py` |
| A crash at any step boundary, or any broker fault, never double-places an order | `tests/chaos/test_crash_matrix.py` |
| The log is append-only and tamper-evident | Postgres triggers + SHA-256 hash chain |
| Money is `Decimal` only | pydantic boundary + AST test |

## Phase 2: running it

```
signal workers ──INSERT only──► intent_inbox ──► EXECUTION SERVICE ──► Alpaca (paper → live)
 (no broker keys)                                 (only process with     Robinhood Crypto (live)
                                                   broker keys)          PaperBroker + live quotes (crypto paper)
                                   event log ◄────────┘  reconciler · alerts (Discord) · /healthz
```

| Account | Broker | Paper | Live |
|---|---|---|---|
| US stocks | Alpaca | Alpaca's real paper endpoint (new account) | same code, live keys |
| Crypto | Robinhood Crypto | `paper-crypto`: simulator on live Robinhood quotes (Robinhood has no sandbox) | `robinhood`: your existing account |

```bash
cp config/example.toml config/local.toml && cp config/crypto_pairs.example.json config/crypto_pairs.json
export ENGINE_DATABASE_URL=postgresql://...      # + ALPACA_PAPER_KEY_ID/SECRET, ROBINHOOD_API_KEY/PRIVATE_KEY,
                                                  #   DISCORD_WEBHOOK_URL (secrets are env-only)
python -m engine.service migrate
python -m engine.service run --config config/local.toml   # curl localhost:8081/healthz
python -m engine.service status --config config/local.toml
python -m engine.service kill --config config/local.toml --account alpaca-paper   # emergency stop: DB only
```

**Before trusting it with real money**, in order:
1. `RUN_LIVE_TESTS=1 pytest -m live_robinhood` (read-only) and `-m live_paper` (Alpaca paper): they check the
   assumptions the fake servers encode.
2. Paper soak for weeks with zero reconciler drift (`DriftDetected` events) and sensible Discord alerts.
3. Going live is three separate deliberate acts: `enabled = true`, `ENGINE_ALLOW_LIVE=1`, and
   `set-mode --mode LIVE --confirm`. Start with a tiny budget and a tight `[accounts.risk]`.

What keeps it safe, beyond the Phase 1 guarantees:

| Guarantee | Where |
|---|---|
| Only the execution service holds broker keys; workers can only INSERT into the inbox (verified against Postgres) | `tests/unit/test_inbox.py` |
| Two services can never trade the same accounts; losing the lock stops trading at once | `tests/unit/test_leader.py`, `test_executor.py` |
| Bad credentials halt the account (kill switch + CRITICAL alert), never retried | `test_executor.py` |
| Every position has a working stop or a pending exit; a rejected stop flattens it and trips the strategy | `test_separate_protection.py` |
| Broker vs ledger drift (positions, cash, orders) is detected; persistent critical drift throws the kill switch | `test_reconciler.py` |
| A retry can never double-place: deterministic client ids (UUIDv5 for Robinhood), reconcile-before-resubmit | `tests/contract`, `tests/chaos` |
| Paper and live reject the same orders (shared validation) and exercise the same OrderManager path | `test_paper_broker.py` |
| Secrets never reach logs, errors or alerts (confined to adapter/service layers; redaction; httpx URL logging off) | `test_invariants.py`, `test_http.py`, `test_alerts.py` |

## Run the tests

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest                      # uses a throwaway local Postgres if `initdb` is on PATH,
                                      # or set TEST_DATABASE_URL to an admin DSN
.venv/bin/ruff check src && .venv/bin/mypy
docker compose up -d postgres         # optional: a persistent dev database
```

## Layout

- `domain/` frozen pydantic models: Intent, events, market snapshot
- `eventlog/` store contract, in-memory and Postgres implementations, migrations
- `projections/` deterministic fold of events into account and order state
- `risk/` layered config (stricter layer wins), individual checks, the engine
- `broker/` adapter contract v2; `MockBroker` (fault injection), `PaperBroker` (persisted simulator), `AlpacaBroker`, `RobinhoodCryptoBroker`, shared HTTP layer
- `marketdata/` quote/volume/asset providers (Alpaca, Robinhood + allowlist); fail closed on bad data
- `service/` executor loop, intent inbox, leader lock, Discord alerts, config/secrets, health, CLI
- `reconciler.py` broker-vs-ledger drift detection with a kill-switch policy
- `orders/` order state machine and `OrderManager` (write-ahead, idempotent, recoverable)
- `control.py` funding, kill switch, paper/live mode (LIVE needs explicit confirmation), breaker resets
- `core.py` `TradingCore`: the single path from Intent to order, plus `recover()`

## Known limits

- Synchronous, single-writer (one executor per database). Throughput is not a goal; correctness is.
- Alpaca: a bracket's legs activate only after the **entry fully fills**, so a partially filled entry is
  unprotected at the broker until it completes (the ledger counts it as covered). Whole shares only.
- Robinhood: only a stop-loss can be held at the broker (no OCO), so **take-profit intents are rejected**
  (`TP_NOT_SUPPORTED`). Fees are modelled as 0 live (spread is in the price) and 0.25% in the simulator.
- Existing Robinhood holdings are imported as `unmanaged` at their market price (true cost basis is not
  available): they count toward exposure and equity but strategies can never sell them. A manual trade in the
  Robinhood app shows up as drift; resolve it with `adjust` after review.
- Realized-P&L-based breakers; unrealized drawdown is not yet a trigger. Breakers latch until reset.
- Alerts are best-effort and start from "now" after a restart (history is in the event log, not replayed).
- The Robinhood signing format and quote field names are implemented from the published spec summary and
  must be confirmed once with the live read-only test before relying on them.
