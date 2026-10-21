"""Offline wire-level regressions: a local client and a mocked Bifrost upstream."""

import base64
import concurrent.futures
import json
import threading
from datetime import date
from functools import partial
from pathlib import Path

import pytest

pytest.importorskip("httpx")
pytest.importorskip("pypdf")


import httpx

from extract_bench.inference.providers.extract.gemini_cli_trace import GeminiCliTrace
from extract_bench.inference.providers.extract.gemini_cli_transport import GeminiCliTransport
from extract_bench.utils.gemini_pricing import cost_from_request_usage as price_request

cost_from_request_usage = partial(price_request, as_of=date(2026, 10, 21))


class Stream(httpx.SyncByteStream):
    def __init__(self, generate):
        self.generate = generate

    def __iter__(self):
        yield from self.generate()


def native_frame(total=1200):
    return (
        "data: "
        + json.dumps(
            {
                "candidates": [{"content": {"parts": [{"text": "évidence"}]}}],
                "usageMetadata": {"promptTokenCount": 1000, "totalTokenCount": total, "cachedContentTokenCount": 400},
            },
            ensure_ascii=False,
        )
        + "\r\n\r\n"
    ).encode()


def journal(trace):
    return [json.loads(line) for line in trace.path.read_text().splitlines()]


def recorder(tmp_path, upstream, cancel=lambda: None):
    trace = GeminiCliTrace(tmp_path, {"example_id": "doc"})
    proxy = GeminiCliTransport(
        "http://gateway.test/genai", trace, cost_from_request_usage, cancel, api_key="configured-key"
    )
    proxy._client.close()
    proxy._client = httpx.Client(transport=httpx.MockTransport(upstream), trust_env=False)
    return trace, proxy


def test_stream_usage_is_durable_before_eof_and_cumulative_updates_are_not_double_counted(tmp_path: Path) -> None:
    """A timeout/spot loss after usage arrives must retain both native response bytes and known cost."""
    usage_seen, release = threading.Event(), threading.Event()
    first, last = native_frame(), native_frame(1300)
    requests = []

    def generate():
        # Arbitrary byte boundaries, including inside UTF-8 and SSE delimiters.
        for byte in first:
            yield bytes([byte])
        assert release.wait(5)
        yield last

    def upstream(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-bf-request-id": "gateway-id", "x-bf-vk": "gateway-secret"},
            stream=Stream(generate),
        )

    trace, proxy = recorder(tmp_path, upstream)
    original_emit = trace.emit

    def emit(event, **payload):
        original_emit(event, **payload)
        if event == "attempt.accounting" and payload["cost_usd"] is not None:
            usage_seen.set()

    trace.emit = emit
    payload = b'{"contents":[{"parts":[{"inlineData":{"mimeType":"application/pdf","data":"Zm9v"}}]}]}'
    with proxy, concurrent.futures.ThreadPoolExecutor() as pool, httpx.Client(trust_env=False) as client:
        future = pool.submit(
            client.post,
            proxy.cli_base_url + "/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse",
            content=payload,
            headers={"x-goog-api-key": "personal-secret", "authorization": "secret"},
        )
        try:
            assert usage_seen.wait(5)
            assert not future.done()
            accounting = [e for e in journal(trace) if e["event"] == "attempt.accounting"][-1]
            assert accounting["cost_usd"] == pytest.approx(0.00324)
            assert accounting["cost_complete"] is False
        finally:
            release.set()
        assert future.result(timeout=5).content == first + last
    proxy.raise_if_failed()
    assert proxy.summary()["total_cost_usd"] == pytest.approx(0.00424)
    assert requests[0].url == "http://gateway.test/genai/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse"
    assert requests[0].content == payload
    assert requests[0].headers["x-goog-api-key"] == "configured-key"
    assert "authorization" not in requests[0].headers
    assert requests[0].headers["x-bf-lh-bench-request-id"] == proxy.summary()["requests"][0]["request_id"]
    assert "personal-secret" not in trace.path.read_text()
    assert "gateway-secret" not in trace.path.read_text()
    chunks = [base64.b64decode(e["data_base64"]) for e in journal(trace) if e["event"] == "model.response.chunk"]
    assert b"".join(chunks) == first + last


def test_failed_stream_cost_survives_retry_without_treating_missing_usage_as_free(tmp_path: Path) -> None:
    """CLI terminal stats omit failed streams; the ledger must retain their observed tokens independently."""
    calls = []

    def fail_after_usage():
        yield native_frame()
        raise httpx.ReadError("stream interrupted")

    def upstream(request):
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Stream(fail_after_usage) if len(calls) == 1 else Stream(lambda: [native_frame(1300)]),
        )

    trace, proxy = recorder(tmp_path, upstream)
    with proxy, httpx.Client(trust_env=False) as client:
        url = proxy.cli_base_url + "/v1beta/models/gemini-4-argon:streamGenerateContent"
        client.post(url, json={})
        client.post(url, json={})
    summary = proxy.summary()
    assert len(summary["requests"]) == 2
    assert summary["known_cost_usd"] == pytest.approx(0.00748)
    assert summary["total_cost_usd"] is None
    assert [item["outcome"] for item in summary["requests"]] == ["error", "success"]
    assert sum(e["event"] == "model.request.started" for e in journal(trace)) == 2


@pytest.mark.parametrize("body", [{"error": {"message": "rejected"}}, {"usageMetadata": {"promptTokenCount": 2}}])
def test_http_error_or_partial_usage_keeps_request_cost_unknown(tmp_path: Path, body) -> None:
    trace, proxy = recorder(tmp_path, lambda request: httpx.Response(400, json=body))
    with proxy, httpx.Client(trust_env=False) as client:
        assert (
            client.post(proxy.cli_base_url + "/v1beta/models/gemini-4-argon:generateContent", json={}).status_code
            == 400
        )
    assert proxy.summary()["known_cost_usd"] is None
    assert proxy.summary()["total_cost_usd"] is None
    assert journal(trace)[-1]["cost_complete"] is False


def test_journal_failure_stops_cli_before_forwarding_paid_request(tmp_path: Path) -> None:
    called, cancelled = [], threading.Event()
    trace, proxy = recorder(tmp_path, lambda request: called.append(request), cancelled.set)
    original = trace.emit

    def emit(event, **payload):
        if event == "model.request.started":
            raise OSError("disk full")
        original(event, **payload)

    trace.emit = emit
    with proxy, httpx.Client(trust_env=False) as client:
        with pytest.raises(httpx.RemoteProtocolError):
            client.post(proxy.cli_base_url + "/v1beta/models/gemini-4-argon:generateContent", json={})
        assert cancelled.wait(5)
        with pytest.raises(RuntimeError, match="capture failed"):
            proxy.raise_if_failed()
    assert called == []


def test_cancellation_retains_observed_usage_and_closes_active_stream_without_changing_timeout(tmp_path: Path) -> None:
    """Stopping a document must retain paid usage and a single cancelled request outcome."""
    pending, release = threading.Event(), threading.Event()

    class PendingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield native_frame()
            pending.set()
            assert release.wait(5)
            raise httpx.ReadError("cancelled")

        def close(self):
            release.set()

    trace, proxy = recorder(
        tmp_path,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=PendingStream(),
        ),
    )
    with concurrent.futures.ThreadPoolExecutor() as pool, httpx.Client(trust_env=False) as client:
        with proxy:
            future = pool.submit(
                client.post, proxy.cli_base_url + "/v1beta/models/gemini-4-argon:streamGenerateContent", json={}
            )
            assert pending.wait(5)
            trace.cancel("per-file timeout")
        future.result(timeout=5)
    summary = proxy.summary()
    assert summary["known_cost_usd"] == pytest.approx(0.00324)
    assert summary["total_cost_usd"] is None
    saved = journal(trace)
    assert [e["reason"] for e in saved if e["event"] == "attempt.cancelled"] == ["per-file timeout"]
    assert [e["outcome"] for e in saved if e["event"] == "model.request.finished"] == ["cancelled"]
    assert summary["requests"][0]["outcome"] == "cancelled"


def test_error_frame_after_usage_does_not_claim_complete_cost(tmp_path: Path) -> None:
    """HTTP 200 is insufficient when the stream subsequently reports a native API error."""
    payload = native_frame() + b'data: {"error":{"code":500,"message":"generation failed"}}\n\n'
    trace, proxy = recorder(
        tmp_path,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload,
        ),
    )
    with proxy, httpx.Client(trust_env=False) as client:
        assert (
            client.post(proxy.cli_base_url + "/v1beta/models/gemini-4-argon:streamGenerateContent", json={}).content
            == payload
        )
    assert proxy.summary()["known_cost_usd"] == pytest.approx(0.00324)
    assert proxy.summary()["total_cost_usd"] is None
    assert proxy.summary()["requests"][0]["outcome"] == "response_error"


@pytest.mark.parametrize("stage", ["headers", "body"])
@pytest.mark.parametrize("tls", [False, True])
def test_cli_disconnect_aborts_gateway_socket_before_headers_or_while_streaming(tmp_path: Path, stage, tls) -> None:
    """The recorder must not leave paid requests running after the CLI drops its local connection."""
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    entered, disconnected, finished, usage_seen = (threading.Event() for _ in range(4))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            if stage == "body":
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.write(native_frame())
                self.wfile.flush()
            entered.set()
            if self.connection.recv(1) == b"":
                disconnected.set()
            self.close_connection = True

        def log_message(self, *_args):
            pass

    gateway = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if tls:
        import ssl
        import subprocess

        cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(cert),
                "-subj",
                "/CN=localhost",
                "-days",
                "1",
            ],
            check=True,
            capture_output=True,
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        gateway.socket = context.wrap_socket(gateway.socket, server_side=True)
    gateway.daemon_threads = True
    thread = threading.Thread(target=gateway.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    trace = GeminiCliTrace(tmp_path, {"example_id": "doc"})
    original = trace.emit

    def emit(event, **payload):
        original(event, **payload)
        if event == "model.request.finished":
            finished.set()
        if event == "attempt.accounting" and payload["cost_usd"] is not None:
            usage_seen.set()

    trace.emit = emit
    try:
        proxy = GeminiCliTransport(
            f"{'https' if tls else 'http'}://127.0.0.1:{gateway.server_port}/genai",
            trace,
            cost_from_request_usage,
            lambda: None,
            api_key="configured-key",
        )
        if tls:
            proxy._client.close()
            proxy._client = httpx.Client(
                verify=False, trust_env=False, timeout=5, limits=httpx.Limits(max_keepalive_connections=0)
            )
        with proxy:
            with socket.create_connection(("127.0.0.1", proxy._server.server_port)) as client:
                client.sendall(
                    b"POST /v1beta/models/gemini-4-argon:streamGenerateContent HTTP/1.1\r\n"
                    b"Host: localhost\r\nContent-Length: 2\r\n\r\n{}"
                )
                assert entered.wait(5)
                if stage == "body":
                    assert usage_seen.wait(5)
                client.shutdown(socket.SHUT_RDWR)
            assert disconnected.wait(5)
            assert finished.wait(5)
            assert proxy.summary()["requests"][0]["outcome"] == "client_disconnected"
            assert proxy.summary()["total_cost_usd"] is None
    finally:
        gateway.shutdown()
        gateway.server_close()
        thread.join()


def test_gateway_error_with_closed_cli_is_not_mislabeled_as_capture_failure(tmp_path: Path, monkeypatch) -> None:
    """A broken pipe sending a gateway error must preserve the ledger and allow subsequent CLI requests."""
    from http.server import BaseHTTPRequestHandler

    calls = []
    cancelled = threading.Event()

    def upstream(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=native_frame())

    def closed_client(*args, **kwargs):
        raise BrokenPipeError("CLI connection is already closed")

    monkeypatch.setattr(BaseHTTPRequestHandler, "send_error", closed_client)
    trace, proxy = recorder(tmp_path, upstream, cancelled.set)
    with proxy, httpx.Client(trust_env=False) as client:
        url = proxy.cli_base_url + "/v1beta/models/gemini-4-argon:streamGenerateContent"
        with pytest.raises(httpx.RemoteProtocolError):
            client.post(url, json={})
        assert client.post(url, json={}).status_code == 200
    proxy.raise_if_failed()
    assert not cancelled.is_set()
    summary = proxy.summary()
    assert len(summary["requests"]) == 2
    assert summary["known_cost_usd"] == pytest.approx(0.00324)
    assert summary["total_cost_usd"] is None
    assert sum(e["event"] == "model.request.finished" for e in journal(trace)) == 2
