"""Opt-in smoke tests against the REAL APIs. They exist to catch the one thing the fake servers cannot: the
fakes being wrong about the real service.

Skipped unless RUN_LIVE_TESTS=1 and the relevant credentials are in the environment.

* live_paper     : Alpaca PAPER account. Places and cancels far-from-market limit orders ($1 AAPL). No real money.
* live_robinhood : Robinhood Crypto, READ-ONLY. Signs requests, reads account/holdings/quotes. Places NOTHING.
* live_news      : Alpaca news, READ-ONLY.
* live_anthropic / live_xai : ONE tiny structured call each (a few hundred tokens). They prove the request shape
                   and response parsing against the real service. Models: LIVE_ANTHROPIC_MODEL / LIVE_XAI_MODEL.
"""

from __future__ import annotations

import os
import uuid
from decimal import Decimal

import pytest
from pydantic import SecretStr

from engine.broker.alpaca import AlpacaBroker
from engine.broker.base import BrokerStatus, OrderRequest
from engine.broker.robinhood_crypto import PairInfo, RobinhoodApi, RobinhoodCryptoBroker
from engine.clock import SystemClock
from engine.domain.types import AssetClass, Mode, Side
from engine.marketdata.alpaca import AlpacaMarketData
from engine.marketdata.robinhood import RobinhoodMarketData

LIVE = os.environ.get("RUN_LIVE_TESTS") == "1"
D = Decimal


def _alpaca() -> AlpacaBroker:
    key, secret = os.environ.get("ALPACA_PAPER_KEY_ID"), os.environ.get("ALPACA_PAPER_SECRET")
    if not (LIVE and key and secret):
        pytest.skip("set RUN_LIVE_TESTS=1, ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET")
    return AlpacaBroker(SecretStr(key), SecretStr(secret), Mode.PAPER)


def _robinhood() -> RobinhoodApi:
    key, priv = os.environ.get("ROBINHOOD_API_KEY"), os.environ.get("ROBINHOOD_PRIVATE_KEY")
    if not (LIVE and key and priv):
        pytest.skip("set RUN_LIVE_TESTS=1, ROBINHOOD_API_KEY and ROBINHOOD_PRIVATE_KEY")
    return RobinhoodApi(SecretStr(key), SecretStr(priv))


@pytest.mark.live_paper
def test_alpaca_paper_account_state_and_clock() -> None:
    broker = _alpaca()
    assert broker.get_cash() >= 0
    assert isinstance(broker.get_positions(), dict)
    assert isinstance(broker.market_clock()["is_open"], bool)
    assert broker.get_asset("AAPL")["tradable"] is True


@pytest.mark.live_paper
def test_alpaca_paper_order_lifecycle_matches_our_assumptions() -> None:
    broker = _alpaca()
    cid = f"engine-smoke-{uuid.uuid4().hex[:12]}"
    req = OrderRequest(
        client_order_id=cid, symbol="AAPL", asset_class=AssetClass.STOCK, side=Side.BUY, quantity=D(1),
        limit_price=D("1.00"), stop_loss_price=D("0.50"),  # resting far below market; OTO because stop only
    )  # fmt: skip
    try:
        first = broker.submit(req)
        assert first.status in (BrokerStatus.ACCEPTED, BrokerStatus.PARTIALLY_FILLED)
        again = broker.submit(req)  # duplicate client_order_id must resolve to the same order
        assert again.broker_order_id == first.broker_order_id
        found = broker.get_order(cid)
        assert found is not None and found.client_order_id == cid
        assert any(o.client_order_id == cid for o in broker.list_open_orders())
        assert broker.get_order(f"{cid}-does-not-exist") is None
    finally:
        broker.cancel(cid)
    after = broker.get_order(cid)
    assert after is not None and after.status in (
        BrokerStatus.CANCELLED,
        BrokerStatus.ACCEPTED,
    )  # cancel may lag


@pytest.mark.live_paper
def test_alpaca_market_data_for_a_liquid_stock() -> None:
    broker = _alpaca()
    key, secret = os.environ["ALPACA_PAPER_KEY_ID"], os.environ["ALPACA_PAPER_SECRET"]
    md = AlpacaMarketData(SecretStr(key), SecretStr(secret), broker, SystemClock())
    i = md.snapshot(["AAPL"]).instruments["AAPL"]
    assert i.price > 0 and i.tradable and i.avg_daily_volume_usd is not None


@pytest.mark.live_robinhood
def test_robinhood_signed_reads_work_and_have_the_shape_we_parse() -> None:
    api = _robinhood()
    pairs = {"BTC-USD": PairInfo(asset_increment=D("0.00000001"), price_increment=D("0.01"))}
    broker = RobinhoodCryptoBroker(api, pairs)
    assert broker.get_cash() >= 0  # accounts/ : signature accepted, buying_power parsed
    assert isinstance(broker.get_positions(), dict)  # holdings/
    assert isinstance(broker.list_open_orders(), list)  # orders/ with pagination
    snap = RobinhoodMarketData(api, pairs, SystemClock()).snapshot(["BTC-USD"])  # best_bid_ask/
    assert snap.instruments["BTC-USD"].price > 0


@pytest.mark.live_paper
def test_alpaca_historical_bars_for_stocks_and_crypto() -> None:
    from datetime import UTC, datetime, timedelta

    from engine.marketdata.alpaca import AlpacaBars

    _alpaca()  # skips unless the paper keys are configured
    key, secret = os.environ["ALPACA_PAPER_KEY_ID"], os.environ["ALPACA_PAPER_SECRET"]
    bars = AlpacaBars(SecretStr(key), SecretStr(secret), SystemClock())
    now = datetime.now(UTC)
    daily = bars.bars("AAPL", "1d", now - timedelta(days=40), now)
    assert len(daily) >= 15 and all(b.high >= b.low for b in daily)
    hourly = bars.bars("BTC-USD", "1h", now - timedelta(days=3), now)  # crypto trades around the clock
    assert len(hourly) >= 48 and hourly == sorted(hourly, key=lambda b: b.ts)
    assert hourly[-1].ts + timedelta(hours=1) <= now  # never the bar still forming


@pytest.mark.live_news
def test_alpaca_news_returns_parseable_items() -> None:
    from datetime import timedelta

    from engine.marketdata.alpaca_news import AlpacaNewsFeed

    key, secret = os.environ.get("ALPACA_PAPER_KEY_ID"), os.environ.get("ALPACA_PAPER_SECRET")
    if not (LIVE and key and secret):
        pytest.skip("set RUN_LIVE_TESTS=1, ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET")
    clock = SystemClock()
    items = AlpacaNewsFeed(SecretStr(key), SecretStr(secret)).fetch(
        clock.now() - timedelta(days=3), ["AAPL", "MSFT", "NVDA"], clock.now()
    )
    assert items, (
        "no news in three days for AAPL/MSFT/NVDA: unexpected, check the account's data subscription"
    )
    first = items[0]
    assert first.headline and first.published_at.tzinfo is not None and first.symbols


def _structured_request():  # type: ignore[no-untyped-def]
    from engine.llm.provider import LLMRequest
    from engine.strategy.llm_schema import DECISIONS_JSON_SCHEMA, SCHEMA_NAME

    return LLMRequest(
        system="You are a cautious analyst. Reply only in the required structure.",
        user='<news_item id="n1">Apple reports record quarterly revenue.</news_item>\n'
        "Universe: AAPL. Position: flat. Decide.",
        schema_name=SCHEMA_NAME,
        json_schema=DECISIONS_JSON_SCHEMA,
        max_output_tokens=400,
    )


def _check_reply(resp) -> None:  # type: ignore[no-untyped-def]
    from engine.strategy.llm_schema import parse_decisions

    assert resp.input_tokens > 0 and resp.output_tokens > 0 and resp.model_reported
    parse_decisions(resp.parsed)  # the real model's output passes OUR strict schema


@pytest.mark.live_anthropic
def test_claude_returns_schema_valid_decisions() -> None:
    from engine.llm.anthropic import AnthropicProvider

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not (LIVE and key):
        pytest.skip("set RUN_LIVE_TESTS=1 and ANTHROPIC_API_KEY")
    model = os.environ.get("LIVE_ANTHROPIC_MODEL", "claude-haiku-5-5")
    _check_reply(AnthropicProvider(SecretStr(key), model).complete(_structured_request()))


@pytest.mark.live_xai
def test_grok_returns_schema_valid_decisions() -> None:
    from engine.llm.xai import XAIProvider

    key = os.environ.get("XAI_API_KEY")
    if not (LIVE and key):
        pytest.skip("set RUN_LIVE_TESTS=1 and XAI_API_KEY")
    model = os.environ.get("LIVE_XAI_MODEL", "grok-4")
    _check_reply(XAIProvider(SecretStr(key), model).complete(_structured_request()))
