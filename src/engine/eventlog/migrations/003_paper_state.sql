-- State of the simulated (paper) broker, so a restart does not forget its orders, cash and positions.
CREATE TABLE paper_broker_state (
    account_id  TEXT        PRIMARY KEY,
    state       JSONB       NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
