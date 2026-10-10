"""Health endpoint: a tiny read-only HTTP server on localhost for container/orchestrator probes."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class HealthServer:
    def __init__(self, port: int, status: Callable[[], dict[str, Any]]) -> None:
        self._status = status
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path not in ("/healthz", "/"):
                    self.send_error(404)
                    return
                body = outer._status()
                ok = bool(body.get("ok"))
                data = json.dumps(body).encode()
                self.send_response(200 if ok else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: object) -> None:  # silence default stderr logging
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
