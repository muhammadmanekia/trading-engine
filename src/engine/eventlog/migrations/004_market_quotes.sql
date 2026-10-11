-- Quotes recorded by the execution service so backtests can use the REAL spread we actually paid/saw, and so a
-- paper/live run can be replayed against what the market looked like at the time. Not part of the audit log.
CREATE TABLE market_quotes (
    id          BIGSERIAL    PRIMARY KEY,
    ts          TIMESTAMPTZ  NOT NULL,
    account_id  TEXT         NOT NULL,
    symbol      TEXT         NOT NULL,
    price       NUMERIC      NOT NULL,
    spread_bps  NUMERIC
);

CREATE INDEX market_quotes_symbol_ts_idx ON market_quotes (symbol, ts);
