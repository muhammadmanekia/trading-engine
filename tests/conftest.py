"""Fixtures. Postgres tests run against a throwaway local cluster (or TEST_DATABASE_URL if provided)."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from engine.eventlog.base import EventStore
from engine.eventlog.memory import InMemoryEventStore
from engine.eventlog.postgres import PostgresEventStore, migrate


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="session")
def pg_admin_dsn(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str | None]:
    if url := os.environ.get("TEST_DATABASE_URL"):
        yield url
        return
    initdb, pg_ctl = shutil.which("initdb"), shutil.which("pg_ctl")
    if not initdb or not pg_ctl:
        yield None
        return
    root: Path = tmp_path_factory.mktemp("pg")
    data = root / "data"
    subprocess.run(
        [initdb, "-D", str(data), "-A", "trust", "-U", "postgres"], check=True, capture_output=True
    )
    port = _free_port()
    opts = f"-c listen_addresses=127.0.0.1 -c port={port} -c unix_socket_directories='' -c fsync=off"
    subprocess.run(
        [pg_ctl, "start", "-D", str(data), "-o", opts, "-w", "-l", str(root / "pg.log")],
        check=True,
        capture_output=True,
    )
    try:
        yield f"postgresql://postgres@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run([pg_ctl, "stop", "-D", str(data), "-m", "immediate"], capture_output=True)


@pytest.fixture
def pg_dsn(pg_admin_dsn: str | None) -> Iterator[str]:
    if pg_admin_dsn is None:
        pytest.skip("no Postgres available (install postgres or set TEST_DATABASE_URL)")
    name = f"t_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    base, _, _ = pg_admin_dsn.rpartition("/")
    dsn = f"{base}/{name}"
    migrate(dsn)
    try:
        yield dsn
    finally:
        with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


@pytest.fixture(params=["memory", "postgres"])
def store(request: pytest.FixtureRequest) -> Iterator[EventStore]:
    """Every contract/integration test runs against both implementations."""
    if request.param == "memory":
        yield InMemoryEventStore()
        return
    dsn: str = request.getfixturevalue("pg_dsn")
    pg = PostgresEventStore(dsn)
    try:
        yield pg
    finally:
        pg.close()
