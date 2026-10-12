-- News archive, LLM call records and trade proposals. These are working/audit tables next to the event log; the
-- authoritative trail of what an LLM decided is the LlmDecisionMade event, which carries hashes of what is stored here.

CREATE TABLE news_items (
    source       TEXT        NOT NULL,
    id           TEXT        NOT NULL,
    headline     TEXT        NOT NULL,
    summary      TEXT        NOT NULL DEFAULT '',
    symbols      JSONB       NOT NULL DEFAULT '[]',
    url          TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    received_at  TIMESTAMPTZ NOT NULL,
    backfilled   BOOLEAN     NOT NULL DEFAULT false,
    PRIMARY KEY (source, id)
);
CREATE INDEX news_items_received_idx ON news_items (received_at);

CREATE TABLE llm_calls (
    call_id         TEXT        PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL,
    account_id      TEXT        NOT NULL,
    strategy_id     TEXT        NOT NULL,
    mode            TEXT        NOT NULL,      -- primary | shadow
    provider        TEXT        NOT NULL,
    model_requested TEXT        NOT NULL,
    model_reported  TEXT,
    prompt_name     TEXT        NOT NULL,
    prompt_version  TEXT        NOT NULL,
    prompt_hash     TEXT        NOT NULL,
    input_hash      TEXT        NOT NULL,
    system_text     TEXT        NOT NULL,
    user_text       TEXT        NOT NULL,
    response_text   TEXT,
    valid           BOOLEAN     NOT NULL,
    error           TEXT,
    decisions       JSONB,
    input_tokens    INTEGER     NOT NULL DEFAULT 0,
    output_tokens   INTEGER     NOT NULL DEFAULT 0,
    cost_usd        NUMERIC     NOT NULL DEFAULT 0,
    latency_ms      INTEGER     NOT NULL DEFAULT 0,
    trigger         TEXT        NOT NULL DEFAULT ''
);
CREATE INDEX llm_calls_input_hash_idx ON llm_calls (input_hash);
CREATE INDEX llm_calls_ts_idx ON llm_calls (ts);

CREATE TABLE proposals (
    proposal_id TEXT        PRIMARY KEY,
    account_id  TEXT        NOT NULL,
    strategy_id TEXT        NOT NULL,
    intent      JSONB       NOT NULL,
    thesis      TEXT        NOT NULL,
    cited       JSONB       NOT NULL DEFAULT '[]',
    status      TEXT        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    created_at  TIMESTAMPTZ NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    decided_at  TIMESTAMPTZ,
    call_id     TEXT
);
