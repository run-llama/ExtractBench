"""Regression coverage for Extend upload response versions."""

from pathlib import Path

import httpx
import pytest

pytest.importorskip("extend_ai", reason="Extend extra required")

from extend_ai import Extend

from extract_bench.inference.providers.extract.extend import ExtendProvider


@pytest.mark.parametrize("api_version", ["2025-04-21", "2026-02-09"])
def test_upload_response_envelope_does_not_turn_missing_sdk_id_into_none_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_version: str
) -> None:
    file = tmp_path / "document.pdf"
    file.write_bytes(b"%PDF-1.7")
    file_response = {
        "id": "file_uploaded",
        "object": "file",
        "name": "document.pdf",
        "metadata": {},
        "createdAt": "2026-11-10T00:00:00Z",
        "updatedAt": "2026-11-10T00:00:00Z",
    }

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/files/upload"
        body = {"success": True, "file": file_response} if api_version == "2025-04-21" else file_response
        return httpx.Response(200, json=body)

    monkeypatch.setenv("EXTEND_API_KEY", "test")
    provider = ExtendProvider("extend", {"api_version": api_version})
    provider._client = Extend(
        token="test",
        extend_api_version=api_version,
        httpx_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    assert provider._upload_file(str(file)) == "file_uploaded"
