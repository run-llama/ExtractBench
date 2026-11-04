"""Record native Gemini HTTP traffic before forwarding it between the CLI and its configured API endpoint.

The CLI's final stats omit failed streams. This loopback adapter performs no
model/protocol translation and never retries: each CLI request gets its own ID,
fsynced request/response events, and independently reconciled usage.
"""

from __future__ import annotations

import base64
import json
import select
import socket
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from extract_bench.inference.providers.extract.gemini_cli_trace import GeminiCliTrace


class _UsageReader:
    """Decode SSE frames across arbitrary HTTP chunk and UTF-8 boundaries."""

    def __init__(self, streaming: bool, observe: Callable[[Mapping[str, Any]], None]) -> None:
        self.streaming = streaming
        self.observe = observe
        self.buffer = b""
        self.data: list[bytes] = []
        self.error: str | None = None

    def feed(self, chunk: bytes) -> None:
        self.buffer += chunk
        if self.streaming:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                line = line.rstrip(b"\r")
                if line.startswith(b"data:"):
                    self.data.append(line[5:].lstrip(b" "))
                elif len(line) == 0:
                    self._frame(b"\n".join(self.data))
                    self.data.clear()

    def _frame(self, data: bytes) -> None:
        if len(data) == 0 or data == b"[DONE]":
            return
        try:
            value = json.loads(data)
        except json.JSONDecodeError as exc:
            self.error = f"Malformed Gemini response: {exc}"
            return
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict):
                if "error" in item:
                    self.error = json.dumps(item["error"])
                if isinstance(item.get("usageMetadata"), dict):
                    self.observe(item)

    def finish(self) -> None:
        if self.streaming:
            # An unterminated SSE frame is not a complete usage report.
            if len(self.buffer.strip()) > 0 or len(self.data) > 0:
                self.error = "Truncated Gemini SSE frame"
        elif len(self.buffer) > 0:
            self._frame(self.buffer)


class GeminiCliTransport:
    def __init__(
        self,
        base_url: str,
        trace: GeminiCliTrace,
        price: Callable[[str, Mapping[str, Any]], float | None],
        cancel_cli: Callable[[], Any],
        *,
        api_key: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.trace = trace
        self.price = price
        self.cancel_cli = cancel_cli
        self._api_key = api_key
        self._headers = dict(headers) if headers is not None else {}
        self._lock = threading.RLock()
        self._requests: dict[str, dict[str, Any]] = {}
        self._responses: set[httpx.Response] = set()
        self._sockets: set[socket.socket] = set()
        self._errors: list[Exception] = []
        self._closed = False
        # A fresh connection exposes its socket via HTTPX's trace extension, so
        # a CLI disconnect can abort an upstream request even before headers arrive.
        self._client = httpx.Client(
            timeout=httpx.Timeout(10, read=None),
            follow_redirects=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                try:
                    owner._handle(self)
                except Exception as exc:
                    # A journal/decoder failure must stop inference, not merely
                    # kill the HTTP worker while the CLI keeps spending.
                    owner._errors.append(exc)
                    owner.cancel_cli()
                    self.close_connection = True

            do_GET = do_POST

            def log_message(self, *_args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> GeminiCliTransport:
        self._thread.start()
        return self

    @property
    def cli_base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def __exit__(self, *_args: Any) -> None:
        try:
            with self._lock:
                self._closed = True
                for request_id, item in self._requests.items():
                    if not item["finished"]:
                        item.update(finished=True, outcome="cancelled", cost_complete=False)
                        self.trace.emit("model.request.finished", request_id=request_id, **item)
                self._account()
        finally:
            with self._lock:
                connections, responses = list(self._sockets), list(self._responses)
            for connection in connections:
                self._shutdown(connection)
            for response in responses:
                response.close()
            self._client.close()
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()

    def raise_if_failed(self) -> None:
        if len(self._errors) > 0:
            raise RuntimeError("Gemini HTTP trace capture failed") from self._errors[0]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            requests = [{"request_id": key, **value} for key, value in self._requests.items() if value["billable"]]
            costs = [item["cost_usd"] for item in requests if item["cost_usd"] is not None]
            complete = len(requests) > 0 and all(item["cost_complete"] for item in requests)
            known = sum(costs) if len(costs) > 0 else None
            return {
                "requests": requests,
                "known_cost_usd": known,
                "total_cost_usd": known if complete else None,
                "cost_complete": complete,
            }

    def _account(self) -> None:
        summary = self.summary()
        self.trace.record_cost(summary["known_cost_usd"], complete=summary["cost_complete"], usage=summary)

    def _observe(self, request_id: str, value: Mapping[str, Any]) -> None:
        with self._lock:
            item = self._requests[request_id]
            previous_usage = item["usage"]
            usage = {**(previous_usage if previous_usage is not None else {}), **value["usageMetadata"]}
            model = value.get("modelVersion")
            if model is None:
                model = item["model"]
            cost = self.price(model, usage)
            item.update(usage=usage, reported_model=model, cost_usd=cost)
            self.trace.emit("model.usage", request_id=request_id, model=model, usage=usage, cost_usd=cost)
            self._account()

    @staticmethod
    def _shutdown(connection: socket.socket) -> None:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            # The request thread may already have closed its socket.
            pass

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        item: dict[str, Any] | None = None
        headers_sent = False
        outcome = "error"
        error = None
        status = None
        done, disconnected = threading.Event(), threading.Event()
        upstream_socket: socket.socket | None = None

        def observe_connection(name: str, info: Mapping[str, Any]) -> None:
            nonlocal upstream_socket
            if name in {"connection.connect_tcp.complete", "connection.start_tls.complete"}:
                connection = info["return_value"].get_extra_info("socket")
                if not isinstance(connection, socket.socket):
                    raise RuntimeError("Gemini gateway transport did not expose its cancellation socket")
                if upstream_socket is not None:
                    with self._lock:
                        self._sockets.discard(upstream_socket)
                upstream_socket = connection
                with self._lock:
                    self._sockets.add(upstream_socket)
                    if self._closed or disconnected.is_set():
                        self._shutdown(upstream_socket)

        def watch_disconnect() -> None:
            while not done.wait(0.1):
                try:
                    ready, _, _ = select.select([handler.connection], [], [], 0)
                    if len(ready) > 0 and handler.connection.recv(1, socket.MSG_PEEK) == b"":
                        disconnected.set()
                        if upstream_socket is not None:
                            self._shutdown(upstream_socket)
                        return
                except OSError:
                    # An unread response can make CLI cancellation reset the
                    # socket rather than return EOF. Both must abort upstream.
                    disconnected.set()
                    if upstream_socket is not None:
                        self._shutdown(upstream_socket)
                    return

        watcher = threading.Thread(target=watch_disconnect, daemon=True)
        try:
            path = urlsplit(handler.path)
            # The SDK uses model endpoints only. Never accept an absolute forwarding URL.
            if path.scheme != "" or path.netloc != "" or not path.path.startswith(("/v1beta/models", "/v1/models")):
                handler.send_error(400, "Expected a native Gemini model endpoint")
                return
            query = urlencode([(k, v) for k, v in parse_qsl(path.query) if k != "key"])
            relative = urlunsplit(("", "", path.path, query, ""))
            body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
            model = path.path.split("/models/", 1)[-1].split(":", 1)[0]
            item = {
                "model": model,
                "billable": path.path.endswith((":generateContent", ":streamGenerateContent")),
                "finished": False,
                "cost_usd": None,
                "cost_complete": False,
                "usage": None,
            }
            with self._lock:
                if self._closed or self.trace.cancelled.is_set() or len(self._errors) > 0:
                    item = None
                    handler.send_error(503, "Extraction cancelled")
                    return
                self._requests[request_id] = item
                self.trace.emit(
                    "model.request.started",
                    request_id=request_id,
                    method=handler.command,
                    url=self.base_url + relative,
                    body_base64=base64.b64encode(body).decode(),
                    model=model,
                )
                self._account()
            headers = {"x-goog-api-key": self._api_key, **self._headers, "accept-encoding": "identity"}
            for name in ("content-type", "accept", "user-agent", "x-goog-api-client"):
                if name in handler.headers:
                    headers[name] = handler.headers[name]
            headers["x-bf-lh-bench-request-id"] = request_id
            headers["x-bf-lh-bench-attempt"] = self.trace.path.name
            watcher.start()
            with self._client.stream(
                handler.command,
                self.base_url + relative,
                headers=headers,
                content=body,
                extensions={"trace": observe_connection},
            ) as response:
                with self._lock:
                    self._responses.add(response)
                status = response.status_code
                self.trace.emit(
                    "model.response.started",
                    request_id=request_id,
                    status=status,
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k
                        in {
                            "x-bf-request-id",
                            "x-bf-trace-id",
                            "x-bf-log-id",
                            "x-bf-provider",
                            "x-bf-model",
                            "content-type",
                        }
                    },
                    time_to_headers_s=time.monotonic() - started,
                )
                handler.send_response(status)
                for name in ("content-type", "retry-after"):
                    if name in response.headers:
                        handler.send_header(name, response.headers[name])
                handler.send_header("Connection", "close")
                handler.end_headers()
                headers_sent = True
                reader = _UsageReader(
                    "text/event-stream" in response.headers.get("content-type", ""),
                    lambda value: self._observe(request_id, value),
                )
                for chunk in response.iter_bytes():
                    self.trace.emit(
                        "model.response.chunk", request_id=request_id, data_base64=base64.b64encode(chunk).decode()
                    )
                    reader.feed(chunk)
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                reader.finish()
                error = reader.error
                outcome = "success" if response.is_success and error is None else "response_error"
        except (httpx.HTTPError, ConnectionError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not headers_sent:
                try:
                    handler.send_error(502, "Gemini gateway request failed; see attempt trace")
                except ConnectionError:
                    # The CLI may already have abandoned this request. Preserve
                    # its transport outcome without treating it as a journal failure.
                    disconnected.set()
        finally:
            done.set()
            if watcher.ident is not None:
                watcher.join()
            if upstream_socket is not None:
                with self._lock:
                    self._sockets.discard(upstream_socket)
            if disconnected.is_set():
                outcome = "client_disconnected"
            if item is not None:
                with self._lock:
                    already_finished = item["finished"]
                    item.update(
                        finished=True,
                        outcome=item["outcome"] if already_finished else outcome,
                        status=status,
                        error=error,
                        duration_s=time.monotonic() - started,
                        cost_complete=outcome == "success" and item["cost_usd"] is not None and not self._closed,
                    )
                    self.trace.emit(
                        "model.request.late_completion" if already_finished else "model.request.finished",
                        request_id=request_id,
                        **item,
                    )
                    self._account()
                    self._responses = {r for r in self._responses if not r.is_closed}
            handler.close_connection = True
