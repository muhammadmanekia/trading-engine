# LLM strategies and the news listener

A language model reads fresh news about a short list of stocks and **proposes** trades. Everything after the
proposal is ordinary, deterministic engine code. This page explains what the model can and cannot do, how to run
it, and how to find out whether it is any good.

## What the model can and cannot do

| The model can | The model cannot |
|---|---|
| say `enter`, `exit` or `hold` for a symbol in the strategy's universe | set a price, a quantity or an order type |
| give a confidence (0 to 1), a horizon, a stop distance as a percentage, and a short thesis | see a broker key, a webhook, or any credential |
| cite which supplied news items it relied on | cite an item it was not given (the decision is discarded) |
| | place an order: it only produces a `Proposal` or an `Intent`; the risk engine still decides |

Quantity comes from confidence tiers in code (`[sizing]`), the stop price is computed by the engine from the
percentage, and the validator throws away anything outside the rules: unknown symbol, fabricated citation,
confidence below `min_confidence`, stop outside `[min_stop_pct, max_stop_pct]`, more entries than
`max_entries_per_day`, entering what is already held, exiting what is not.

## Prompt injection

News text is written by strangers. Defence is layered, and only the last layer is relied upon:

1. **Sanitise:** Unicode-normalise, strip invisible and control characters, replace `<` and `>`, truncate.
2. **Screen:** headlines matching known attack shapes (ignore-previous-instructions, role-play, chat markers, tool
   syntax, "reveal your prompt", our own delimiters, encoded payloads) never reach a model. A human gets a
   Discord warning and the item is archived as evidence.
3. **Tag:** the prompt wraps each item in `<news_item>` tags and tells the model it is untrusted data.
4. **Structure (the real boundary):** even a model that obeys every attack can only emit the small schema above,
   and the validator and risk engine bound the result. `tests/llm/test_source.py` runs a scripted "obedient
   attacker" model that tries every lever at once and asserts no out-of-bounds trade results.
5. **Human:** `propose_only` mode (the default for new strategies) means nothing trades until you approve it.

## Running it

```bash
export ANTHROPIC_API_KEY=...   XAI_API_KEY=...           # only the providers you use
# 1. add the [llm] section from config/example.toml, and under your Alpaca account:
#      llm_strategies = ["strategies/examples/news-swing-stocks.toml"]
python -m engine.service run --config config/local.toml

python -m engine.service proposals --config config/local.toml            # what is waiting for you
python -m engine.service approve   --config config/local.toml --id <proposal id>
python -m engine.service reject    --config config/local.toml --id <proposal id>
python -m engine.service llm-report --config config/local.toml --days 30 --horizon-bars 24
```

Proposals appear in Discord with the thesis, the cited headlines and the exact commands. They expire (default 30
minutes) because stale news is a bad reason to trade. Approving queues a fresh intent with a 10-minute lifetime and
the risk engine judges it against the market as it is *then*.

Headlines that arrive while the market is closed wait for the open if they are still younger than
`news_max_age_minutes`, otherwise they are archived and ignored.

### Modes (`[models] mode`)

* `propose_only`: model idea becomes a proposal for you to approve. Start here.
* `auto`: the validated idea becomes an intent immediately (the risk engine still decides). Only after weeks of
  evidence from `llm-report`.
* `shadow_only`: nothing trades; every model is asked and recorded for scoring. A good way to compare models
  with zero risk.

With `shadow = ["grok"]` the extra models are asked the identical question and recorded, but never act.
`consensus = true` makes the strategy act only when every listed model agrees (using the lowest confidence); a
missing or failed voter means no action.

## Cost control

* Daily and monthly dollar caps in `[llm]`, plus per-strategy `daily_usd` and `max_calls_per_day`.
* The worst-case cost (all input tokens plus the full output allowance) is checked **before** each call.
* Identical questions (same prompt version, same inputs) are answered from the record at no cost.
* An unpriced provider is charged at a pessimistic default so a missing price makes the bot more cautious.
* 80% of a cap sends a Discord warning; reaching it simply stops calls.

## Audit trail

Every call is stored in `llm_calls` (full system prompt, user data, raw reply, model requested *and* reported,
tokens, cost, latency, prompt name/version/hash). Every resulting decision appends a hash-chained
`LlmDecisionMade` event with the call id and hashes, so you can always answer "why did it do that?". Prompts
live in `prompts/<name>/v<N>.md`; a new version is a new file, and each call records which one it used.

## Is it any good? (honest evaluation)

You cannot backtest a news model on history the model was trained on: it may simply remember what happened. So:

* The news archive records `received_at` from day one, so future replays use what was actually known when.
* `llm-report` scores each recorded *enter* decision **forward**: entry at the next bar's open, exit
  `--horizon-bars` later, compared with the average return of every same-length window for that symbol (so a
  rising stock does not make every buy look clever), with a bootstrap interval on the excess return.
* Fewer than 30 scored decisions gets no verdict, only "anecdote".
* Decisions about periods before a model's `knowledge_cutoff` are counted but excluded from the headline numbers.
* The same report compares the models: invalid-reply rate, cost, latency, and how often they reached the same
  conclusion on the same trigger.

Expect to wait weeks before the numbers mean anything. If the interval straddles zero, the model has shown no edge.

## Not built yet

* Phase 6 learning from the bot's own trades: the prompt has a "LEARNING DIGEST (none yet)" slot ready for it.
* A news WebSocket stream (polling every 30 seconds is plenty for swing trades and much simpler to keep correct).
* Crypto news strategies (stocks first, as agreed).
* Replay of archived news through a model for offline experiments.
