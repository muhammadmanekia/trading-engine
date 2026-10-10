# trading-engine

Event-sourced, fail-closed trading core. **Phase 1:** event log, risk engine, order manager, mock broker.
No real broker, no network, no credentials yet (those arrive in Phase 2).

Design: [`../docs/superpowers/specs/2026-10-10-trading-bot-design.md`](../docs/superpowers/specs/2026-10-10-trading-bot-design.md)

```
Intent ──► Risk engine ──► Order manager ──► Broker (MockBroker now)
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

## Run

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
- `broker/` adapter contract and `MockBroker` with fault injection
- `orders/` order state machine and `OrderManager` (write-ahead, idempotent, recoverable)
- `control.py` funding, kill switch, paper/live mode (LIVE needs explicit confirmation), breaker resets
- `core.py` `TradingCore`: the single path from Intent to order, plus `recover()`

## Known limits (deliberate, for later phases)

- Synchronous, single-writer. Phase 2 wraps it in the execution service process.
- Protective legs appear in the ledger once the entry's size is final; during a partial fill the
  position is protected only at the broker (if the broker supports it).
- Realized-P&L-based breakers; unrealized drawdown is not yet a trigger.
- Breakers latch until `Control.reset_breaker` (no automatic next-day reset).
