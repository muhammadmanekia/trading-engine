CREATE TABLE events (
    seq              BIGSERIAL PRIMARY KEY,
    event_id         UUID        NOT NULL UNIQUE,
    stream_id        TEXT        NOT NULL,
    stream_version   INTEGER     NOT NULL CHECK (stream_version >= 0),
    type             TEXT        NOT NULL,
    account_id       TEXT        NOT NULL,
    schema_version   INTEGER     NOT NULL DEFAULT 1,
    payload          JSONB       NOT NULL,
    ts               TIMESTAMPTZ NOT NULL,
    idempotency_key  TEXT        UNIQUE,
    prev_hash        CHAR(64)    NOT NULL,
    hash             CHAR(64)    NOT NULL UNIQUE,
    UNIQUE (stream_id, stream_version)
);

CREATE INDEX events_account_seq_idx ON events (account_id, seq);

-- The log is append-only: no updates, deletes or truncates, ever.
CREATE FUNCTION events_reject_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'events table is append-only (% not permitted)', TG_OP;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER events_no_update_delete
    BEFORE UPDATE OR DELETE ON events
    FOR EACH ROW EXECUTE FUNCTION events_reject_mutation();

CREATE TRIGGER events_no_truncate
    BEFORE TRUNCATE ON events
    FOR EACH STATEMENT EXECUTE FUNCTION events_reject_mutation();
