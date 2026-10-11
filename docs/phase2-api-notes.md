# Phase 2 API notes (spike results)

What was checked against official documentation before writing the adapters, what was *not* verifiable, and
how the code handles each gap. "Verified" = read from the vendor's published docs/OpenAPI during the spike.

## Alpaca (stocks)

| Fact | Status | Consequence in code |
|---|---|---|
| Bracket orders need **both** `take_profit` and `stop_loss` legs | Verified | Entry with only a stop uses `order_class: "oto"` (`broker/alpaca.py::_order_body`) |
| Fractional shares: market/limit/stop/stop-limit allowed but **`day` TIF only**, can't be replaced | Verified | Whole shares only; `qty_step = 1` from market data; fractional rejected locally |
| `client_order_id` ≤ 128 chars, optional, unique | Verified (length); duplicate behaviour **not documented** | Assumed `422` mentioning `client_order_id`; adapter then fetches the existing order. Covered by the live paper test |
| Fetch by client id: `GET /v2/orders:by_client_order_id?client_order_id=` | Verified | `AlpacaBroker.get_order`. Not-found status undocumented: `404` assumed, live test confirms |
| Order object carries `filled_qty`, `filled_avg_price`, not individual executions | Verified | Cumulative fill accounting in `OrderManager._new_fills` |
| Bracket legs activate only after the entry **fully** fills; cancelling one leg cancels the rest; partial TP reduces the stop | Verified | `protection = NATIVE_BRACKET`; legs enter the ledger once the entry is terminal; protection invariant treats an in-flight bracket entry as covered |
| Legs get broker-generated ids | Verified (`legs` array) | Re-keyed to `<client_order_id>:stop` / `:tp` by leg `type` |
| `403` = insufficient buying power/shares; `422` = invalid params; `401` = bad keys | Verified (403/422); 401 standard | 403 -> `BrokerRejected(INSUFFICIENT_FUNDS)`; 401 -> `CredentialsError` (fatal) |
| `GET /v2/clock` for market hours | Standard | `MARKET_CLOSED` risk rule for entries |
| Free IEX feed volume is a small share of consolidated volume | Known | `volume_scale` option + lower `min_avg_daily_volume_usd` in config |

## Robinhood Crypto

| Fact | Status | Consequence in code |
|---|---|---|
| Base `https://trading.robinhood.com`; headers `x-api-key`, `x-timestamp` (unix s), `x-signature` | Verified | `RobinhoodApi` |
| Signature = base64 Ed25519 over `api_key + timestamp + path + method + body` | **Confirmed live (read-only).** Originally partly verified: order of fields from the spec summary; **separator/body serialisation and whether the query string is part of `path` are not stated** | Implemented as plain concatenation with the query string included; `tests/live/test_live_smoke.py::test_robinhood_signed_reads_work...` is the check. A wrong guess yields `401` -> fatal, never a wrong trade |
| Signed requests valid ~30 s | Third-party client docs | Fake server enforces 30 s; `401` is treated as fatal (not retried) |
| `client_order_id` is a **UUID**; "a repeat with the same value returns the original order" | Verified (OpenAPI) | `wire_id()` = uuid5 of our order id. Retries are idempotent by construction |
| Order types: `market`, `limit`, `stop_loss`, `stop_limit`; `time_in_force` only `gtc` | Verified | Entries market/limit, protection `stop_loss` |
| **No bracket / OCO** | Verified (absent from spec) | `protection = SEPARATE_ORDERS`: OrderManager places a stop after the entry fills |
| Two sells can't reserve the same coins (holdings expose `quantity_available_for_trading`) | Inferred from the holdings schema | **Take-profit is rejected** (`TP_NOT_SUPPORTED`) on Robinhood; only a stop is held at the broker |
| Order object has **no ordered-quantity field**, executions have **no id and no fee** | Verified | Fill id = `<wire>:<index>:<timestamp>:<qty>`; fee 0 (spread is inside `effective_price`); quantity remembered from the submit |
| **No lookup by client id**; list supports `state`, `symbol`, `created_at_start/end`, `limit`, `cursor` | Verified | Remember RH ids; otherwise scan `open`/`partially_filled`, then the last 24 h; cursor pagination |
| States: `open`, `canceled`, `partially_filled`, `filled`, `failed` | Verified | Mapped in `_to_order` |
| Holdings: `asset_code`, `total_quantity`, `quantity_available_for_trading`; account: `buying_power`, `buying_power_currency` | Verified | `get_positions` uses `total_quantity` (includes coins reserved by open sells) |
| `v1` API supports only non-fee-tier orders | Vendor help page | Fee modelled as 0 on live; paper simulator charges a configurable 0.25% |
| **No sandbox / paper environment** | Verified (none documented) | Crypto paper mode = `PaperBroker` fed by live quotes, with persisted state and the same order rules (`check_order_against_pairs`) |
| `best_bid_ask` and `trading_pairs` response field names | **Not verifiable from the spec summary** | Quote parser accepts a few candidate field names and **fails closed** on anything else. Increments, caps and liquidity come from the operator-supplied allowlist file (`config/crypto_pairs.json`), which also restricts trading to approved coins |
| Rate limits | Unverified (429 documented, numbers not) | Client-side token bucket (1.5 req/s, burst 30); 429 -> unavailable |

## Live confirmation (run by the owner)

* **Alpaca paper** (`-m live_paper`, 3 tests): PASSED. Confirmed: duplicate `client_order_id` resolves to the same
  order, lookup by client id works and unknown ids return not-found, a stop-only entry is accepted as an OTO
  order, cancel works, quotes/tradability/clock parse.
* **Robinhood Crypto** (`-m live_robinhood`, read-only): PASSED. Confirmed: the request signature format
  (including the query string in the signed path), account `buying_power`, holdings, paginated open-orders
  listing, and that the live `best_bid_ask` response is accepted by the quote parser.
* **Still not exercised against the real service:** placing/cancelling a Robinhood order (deliberately not
  automated: it would use real money), the exact 4xx bodies Robinhood returns for rejected orders, and the
  real Alpaca bracket (two-leg) path. These are covered only by the fake servers until a tiny paper/live order
  is observed.

## How the gaps get closed

1. Run `RUN_LIVE_TESTS=1 pytest -m live_robinhood` with real Robinhood credentials. It is **read-only**
   (account, holdings, orders list, one quote request) and will fail loudly if the signature format or response
   shapes differ from the assumptions above.
2. Run `RUN_LIVE_TESTS=1 pytest -m live_paper` with an Alpaca **paper** key. It places and cancels a $1 limit
   order and confirms duplicate-id behaviour, by-client-id lookup and OTO acceptance.
3. Only after both pass should an account move from the simulator to a real broker.
