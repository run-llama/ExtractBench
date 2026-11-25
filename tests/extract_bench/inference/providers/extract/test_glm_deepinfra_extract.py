from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("PIL", reason="runners extra required")
pytest.importorskip("pdf2image", reason="runners extra required")
pytest.importorskip("pypdf", reason="runners extra required")
pytest.importorskip("openai", reason="runners extra required")

from PIL import Image

from extract_bench.inference.providers.extract.glm_zai_extract import GLMDeepInfraExtractProvider


def test_pdf_rasterization_preserves_every_page_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = GLMDeepInfraExtractProvider("glm_deepinfra_extract", {"api_key": "k", "dpi": 144})
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%PDF-fake")
    calls: list[tuple[str, int]] = []

    def fake_convert(path: str, *, dpi: int) -> list[Image.Image]:
        calls.append((path, dpi))
        return [Image.new("RGB", (4, 4), "white"), Image.new("RGB", (4, 4), "black")]

    monkeypatch.setattr("pdf2image.convert_from_path", fake_convert)

    blocks = provider._build_file_blocks(pdf)

    assert calls == [(str(pdf), 144)]
    assert len(blocks) == 2
    assert all(block["type"] == "image_url" for block in blocks)
    pixels = []
    for block in blocks:
        prefix, payload = block["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/png;base64"
        with Image.open(io.BytesIO(base64.b64decode(payload))) as page:
            pixels.append(page.getpixel((0, 0)))
    assert pixels == [(255, 255, 255), (0, 0, 0)]


def test_chat_request_includes_image_and_schema_and_decodes_json(tmp_path: Path) -> None:
    provider = GLMDeepInfraExtractProvider("glm_deepinfra_extract", {"api_key": "k"})
    image = tmp_path / "document.png"
    Image.new("RGB", (4, 4), "white").save(image)
    calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps({"invoice_number": "INV-001"})),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                prompt_tokens_details=SimpleNamespace(cached_tokens=10),
            ),
        )

    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    result = provider._call_api(
        {"type": "object", "properties": {"invoice_number": {"type": "string"}}},
        image,
    )

    assert len(calls) == 1
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert [part["type"] for part in calls[0]["messages"][1]["content"]] == ["image_url", "text"]
    content = calls[0]["messages"][1]["content"]
    assert base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1]) == image.read_bytes()
    assert '"invoice_number"' in content[1]["text"]
    assert result["data"] == {"invoice_number": "INV-001"}
