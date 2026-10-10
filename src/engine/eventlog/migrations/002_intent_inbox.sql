-- Signal workers hold NO broker keys and cannot write to the event log. They may only INSERT into this
-- inbox; the execution service validates each row against the Intent schema before appending it to the log.
CREATE TABLE intent_inbox (
    seq          BIGSERIAL   PRIMARY KEY,
    intent_id    TEXT        NOT NULL UNIQUE,
    payload      JSONB       NOT NULL,
    received_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    status       TEXT        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'invalid')),
    detail       TEXT
);

CREATE INDEX intent_inbox_pending_idx ON intent_inbox (seq) WHERE status = 'pending';

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'intent_producer') THEN
        CREATE ROLE intent_producer NOLOGIN;
    END IF;
END
$$;

GRANT INSERT ON intent_inbox TO intent_producer;
GRANT USAGE ON SEQUENCE intent_inbox_seq_seq TO intent_producer;
