"""Re-export of the backtest command line (it lives in the service layer, which owns credentials)."""

from engine.service.backtest_cli import main

__all__ = ["main"]
