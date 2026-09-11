"""DeepSeek one-shot structured extraction provider.

DeepSeek-V4.1-Flash (served as ``deepseek-flash``) extraction over the source
document, through DeepSeek's OpenAI-compatible chat completions endpoint. Its
image input accepts JPEG/PNG/GIF/WebP only -- there is no PDF content block --
so the PDF is rasterized once and every page image rides in a single request.

Structured output is ``response_format={"type": "json_object"}`` with the target
JSON schema in the prompt, the same approach as the z.ai GLM one-shot this
subclasses. DeepSeek has no strict ``json_schema`` response format, so the
schema is guidance rather than an enforced grammar; the prompt already names
JSON, which DeepSeek's JSON mode requires.

Thinking is on by default and is switched off here with
``thinking={"type": "disabled"}``, which the OpenAI SDK carries in
``extra_body``. Non-thinking mode is what makes this a genuine one-shot: the
model answers with the JSON directly instead of spending the ``max_tokens``
budget on a chain of thought first.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from extract_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.direct_model_utils import IMAGE_EXTENSIONS
from extract_bench.inference.providers.extract.glm_zai_extract import GLMZaiExtractProvider
from extract_bench.inference.providers.registry import register_provider

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# DeepSeek peak pricing: USD per million tokens (input cache miss, input cache
# hit, output). Off-peak rates are half of these; peak is used so the benchmark
# cost does not swing with the hour a run happens to start.
# Source: https://api-docs.deepseek.com/quick_start/pricing
_DEEPSEEK_EXTRACT_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    "deepseek-flash": (0.30, 0.006, 1.20),
}


@register_provider("deepseek_oneshot_extract")
class DeepSeekOneshotExtractProvider(GLMZaiExtractProvider):
    """One-shot DeepSeek-V4.1-Flash extraction over rendered page images."""

    DEFAULT_MODEL = "deepseek-flash"
    # DeepSeek accepts up to 384K output tokens. Non-thinking mode defaults to
    # 8K, which the long-list schemas blow through -- a max_tokens stop is a
    # hard error here, not a silent truncation.
    DEFAULT_MAX_TOKENS = 65536

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        cfg = dict(base_config or {})
        api_key = cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "DeepSeek API key is required. Set DEEPSEEK_API_KEY or pass api_key in base_config."
            )
        cfg["api_key"] = api_key
        cfg.setdefault("model", self.DEFAULT_MODEL)
        cfg.setdefault("base_url", _DEEPSEEK_BASE_URL)
        cfg.setdefault("max_tokens", self.DEFAULT_MAX_TOKENS)
        super().__init__(provider_name, cfg)

        self._dpi = int(self.base_config.get("dpi", 150))
        max_pages_cfg = self.base_config.get("max_pages")
        self._max_pages: int | None = int(max_pages_cfg) if max_pages_cfg is not None else None

        thinking = self.base_config.get("thinking", "disabled")
        if thinking not in ("enabled", "disabled"):
            raise ProviderConfigError(f"Invalid thinking '{thinking}'. Must be 'enabled' or 'disabled'.")
        self._thinking: str = thinking

    @staticmethod
    def _pricing_for_model(model: str) -> tuple[float, float, float]:
        matches = [
            (prefix, rates) for prefix, rates in _DEEPSEEK_EXTRACT_PRICING_PER_M.items() if model.startswith(prefix)
        ]
        return max(matches, key=lambda item: len(item[0]))[1] if matches else (0.0, 0.0, 0.0)

    @staticmethod
    def _vendor_label() -> str:
        return "DeepSeek"

    def _extra_request_kwargs(self) -> dict[str, Any]:
        return {"extra_body": {"thinking": {"type": self._thinking}}}

    def _build_file_blocks(self, source_path: Path) -> list[dict[str, Any]]:
        ext = source_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            mime = IMAGE_EXTENSIONS[ext]
            b64 = base64.standard_b64encode(source_path.read_bytes()).decode("utf-8")
            return [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]
        if ext != ".pdf":
            raise ProviderPermanentError(
                f"deepseek_oneshot_extract supports PDFs and {set(IMAGE_EXTENSIONS)}, got {source_path.suffix}"
            )

        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProviderPermanentError("pdf2image is required for deepseek_oneshot_extract.") from e

        try:
            images = convert_from_path(str(source_path), dpi=self._dpi)
        except Exception as e:
            raise ProviderPermanentError(f"Error converting PDF to images: {e}") from e
        if not images:
            raise ProviderPermanentError(f"No pages found in PDF: {source_path}")
        if self._max_pages is not None:
            images = images[: self._max_pages]

        blocks: list[dict[str, Any]] = []
        for image in images:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        return blocks

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        # DeepSeek reports the cache split as its own ``prompt_cache_hit_tokens``
        # alongside the OpenAI-compatible ``prompt_tokens_details.cached_tokens``
        # the base class reads. Prefer the native field when present so the
        # cache-hit credit in _estimate_cost_usd is applied on either shape.
        usage_dict = GLMZaiExtractProvider._extract_usage(response)
        usage = getattr(response, "usage", None)
        hit = getattr(usage, "prompt_cache_hit_tokens", None) if usage is not None else None
        if hit is not None:
            usage_dict["cached_input_tokens"] = int(hit or 0)
        return usage_dict

    def _call_api(self, schema: dict[str, Any], source_path: Path) -> dict[str, Any]:
        result = super()._call_api(schema, source_path)
        self._reject_degenerate_json(result.get("data"), schema)
        result["_config"].update(
            {
                "provider": "deepseek",
                "input_mode": "page_images",
                "dpi": self._dpi,
                "max_pages": self._max_pages,
                "thinking": self._thinking,
            }
        )
        return result

    @staticmethod
    def _reject_degenerate_json(data: Any, schema: dict[str, Any]) -> None:
        """Retry DeepSeek's documented degenerate JSON-mode response.

        DeepSeek warns that JSON Output "may occasionally return empty content".
        In practice it answers with a well-formed but contentless object that
        echoes the request shape -- ``{"type": "json_object"}`` -- roughly one
        call in three on some documents. That parses fine, so without this check
        it is recorded as a successful extraction of nothing and silently scores
        as a zero rather than as an error.

        A real answer shares at least one top-level key with the schema. Nothing
        in common means the model never attempted the task, which is transient:
        the identical request succeeds on retry.
        """
        expected = set(schema.get("properties") or {})
        if not expected or not isinstance(data, dict):
            return
        if set(data) & expected:
            return
        raise ProviderTransientError(
            f"DeepSeek returned a contentless JSON-mode response (keys {sorted(data)[:5]}); retrying."
        )

    def _pricing_snapshot(self) -> dict[str, Any]:
        return {
            "pricing_basis": "deepseek_flash_peak",
            "input_price_per_1m": self._input_price_per_1m,
            "cached_input_price_per_1m": self._cached_input_price_per_1m,
            "output_price_per_1m": self._output_price_per_1m,
        }
