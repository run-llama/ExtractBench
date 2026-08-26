"""Kimi K3 one-shot vision extraction provider (Fireworks).

Kimi K3 is a vision-language model served through Fireworks' OpenAI-compatible
chat completions API. This provider rasterizes the document to one image per
page and sends the pages directly to the model in a single call -- no separate
parse stage -- mirroring the other ``*_oneshot_structured_output_file`` VLM
pipelines.

Structured output uses ``response_format={"type": "json_object"}`` with the
extract schema inlined into the prompt. Kimi K3 does not honor the json_schema
guided-decoding form on Fireworks (it emits free-form reasoning instead of the
constrained object), so json_object -- syntactic-JSON enforcement plus
schema-in-prompt -- is the reliable path, the same choice the gemma4 vLLM
one-shot pipeline makes. Fireworks also rejects PDF file inputs, so rasterizing
to page images is the only one-shot path.

The API key is resolved from FIREWORKS_API_KEY first, falling back to
DEEPSEEK_API_KEY (the existing Fireworks-backed key).
"""

from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.direct_model_utils import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_INSTRUCTION,
    IMAGE_EXTENSIONS,
    add_additional_properties_false,
    normalize_extract_result,
    page_count,
    promote_repeated_structure,
)
from extract_bench.inference.providers.extract.vllm_extract import (
    prune_excess_depth,
    salvage_truncated_json,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_FIREWORKS_KIMI_K3_MODEL = "accounts/fireworks/models/kimi-k3"

# USD per million tokens: input, cached input, output.
# Source: Fireworks serverless pricing for kimi-k3 ($3.00 / $0.30 / $15.00).
_KIMI_EXTRACT_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    _FIREWORKS_KIMI_K3_MODEL: (3.00, 0.30, 15.00),
}


@register_provider("kimi_extract")
class KimiExtractProvider(Provider):
    """One-shot vision extraction through Kimi K3 on Fireworks."""

    DEFAULT_MODEL = _FIREWORKS_KIMI_K3_MODEL

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._model: str = self.base_config.get("model", self.DEFAULT_MODEL)
        self._base_url: str = self.base_config.get("base_url", _FIREWORKS_BASE_URL)
        self._api_key = (
            self.base_config.get("api_key") or os.getenv("FIREWORKS_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        )
        if not self._api_key:
            raise ProviderConfigError(
                "FIREWORKS_API_KEY (or DEEPSEEK_API_KEY as a fallback) is required for kimi_extract."
            )

        self._dpi: int = int(self.base_config.get("dpi", 150))
        max_pages_cfg = self.base_config.get("max_pages")
        self._max_pages: int | None = int(max_pages_cfg) if max_pages_cfg is not None else None
        self._max_tokens: int = int(self.base_config.get("max_tokens", 32768))
        self._temperature: float = float(self.base_config.get("temperature", 0.0))
        # Kimi K3 is a reasoning model; its thinking tokens count against
        # max_tokens (they are part of completion_tokens), so heavy reasoning
        # eats into the JSON budget. Fireworks accepts reasoning_effort
        # ("none" disables thinking entirely, "low"/"medium"/"high" scale it).
        self._reasoning_effort: str | None = self.base_config.get("reasoning_effort")
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._system_prompt: str = self.base_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._user_instruction: str = self.base_config.get("user_instruction", DEFAULT_USER_INSTRUCTION)
        self._timeout_s: float = float(self.base_config.get("timeout_s", self.base_config.get("api_timeout_s", 900.0)))
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd")

        default_input, default_cached_input, default_output = self._pricing_for_model(self._model)
        self._input_price_per_1m: float = float(self.base_config.get("input_price_per_1m", default_input))
        self._cached_input_price_per_1m: float = float(
            self.base_config.get("cached_input_price_per_1m", default_cached_input)
        )
        self._output_price_per_1m: float = float(self.base_config.get("output_price_per_1m", default_output))

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout_s)

    @staticmethod
    def _pricing_for_model(model: str) -> tuple[float, float, float]:
        matches = [(prefix, rates) for prefix, rates in _KIMI_EXTRACT_PRICING_PER_M.items() if model.startswith(prefix)]
        return max(matches, key=lambda item: len(item[0]))[1] if matches else (0.0, 0.0, 0.0)

    # -- internals ----------------------------------------------------------

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        schema = promote_repeated_structure(schema)
        if self._additional_properties_false:
            return add_additional_properties_false(schema)
        return schema

    def _render_page_images(self, source_path: Path) -> list[str]:
        """Return a list of base64-encoded PNGs, one per (capped) page."""
        ext = source_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return [base64.b64encode(source_path.read_bytes()).decode("utf-8")]
        if ext != ".pdf":
            raise ProviderPermanentError(
                f"kimi_extract supports PDFs and {set(IMAGE_EXTENSIONS)}, got {source_path.suffix}"
            )
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProviderPermanentError("pdf2image is required for kimi_extract.") from e

        try:
            images = convert_from_path(str(source_path), dpi=self._dpi)
        except Exception as e:
            raise ProviderPermanentError(f"Error converting PDF to images: {e}") from e
        if not images:
            raise ProviderPermanentError(f"No pages found in PDF: {source_path}")
        if self._max_pages is not None:
            images = images[: self._max_pages]

        out: list[str] = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return out

    def _build_user_content(self, page_images: list[str], schema: dict[str, Any]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in page_images
        ]
        text = (
            "Extract every field from the attached document page image(s) according to the JSON schema below.\n\n"
            "JSON schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Return only one valid JSON object matching the schema. Do not wrap it in markdown fences.\n\n"
            f"{self._user_instruction}"
        )
        content.append({"type": "text", "text": text})
        return content

    def _call_api(self, schema: dict[str, Any], page_images: list[str]) -> dict[str, Any]:
        # Build the request as a dict and splat it in: the OpenAI SDK's typed
        # create() overloads reject our list/dict message + response_format shapes
        # under strict mypy (same approach as vllm_extract / deepseek_extract).
        kwargs: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._build_user_content(page_images, schema)},
            ],
            "response_format": {"type": "json_object"},
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        content = getattr(choice.message, "content", "") or ""

        truncated = False
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # A reasoning model that hits max_tokens (finish_reason == "length")
            # returns syntactically incomplete JSON. Salvage the completed fields
            # -- a truncated extraction earns partial credit, which beats failing
            # the whole document.
            data = salvage_truncated_json(content)
            if data is None:
                raise ProviderPermanentError(
                    f"Kimi returned unparseable output (finish_reason={finish_reason}, chars={len(content)}): {e}"
                ) from e
            truncated = True
        if finish_reason == "length":
            truncated = True
        data, deep_pruned = prune_excess_depth(data)
        if deep_pruned:
            truncated = True

        return {
            "data": data,
            "truncated": truncated,
            "finish_reason": finish_reason,
            "model": self._model,
            "usage": self._extract_usage(response),
            "_config": {
                "provider": "kimi",
                "base_url": self._base_url,
                "dpi": self._dpi,
                "max_pages": self._max_pages,
                "reasoning_effort": self._reasoning_effort,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "additional_properties_false": self._additional_properties_false,
                "max_cost_usd": self._max_cost_usd,
                **self._pricing_snapshot(),
            },
        }

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
        output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
        total_tokens = getattr(usage, "total_tokens", None) or input_tokens + output_tokens
        cached_input_tokens = getattr(usage, "prompt_cache_hit_tokens", None) or 0
        for details_name in ("prompt_tokens_details", "input_tokens_details"):
            details = getattr(usage, details_name, None)
            if details is None:
                continue
            cached_input_tokens = (
                getattr(details, "cached_tokens", None)
                or getattr(details, "cached_input_tokens", None)
                or cached_input_tokens
            )
        return {
            "input_tokens": int(input_tokens),
            "cached_input_tokens": int(cached_input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }

    def _estimate_extract_cost_usd(self, usage: dict[str, int]) -> float:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
        return (
            uncached_input_tokens / 1_000_000 * self._input_price_per_1m
            + cached_input_tokens / 1_000_000 * self._cached_input_price_per_1m
            + output_tokens / 1_000_000 * self._output_price_per_1m
        )

    def _pricing_snapshot(self) -> dict[str, Any]:
        return {
            "pricing_basis": "kimi_k3_serverless",
            "input_price_per_1m": self._input_price_per_1m,
            "cached_input_price_per_1m": self._cached_input_price_per_1m,
            "output_price_per_1m": self._output_price_per_1m,
        }

    # -- Provider interface -------------------------------------------------

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"{type(self).__name__} only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override."
            )

        started_at = datetime.now()
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        schema = self._prepare_schema(request.schema_override)

        try:
            page_images = self._render_page_images(file_path)
            raw_output = self._call_api(schema, page_images)
        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
            raise
        except Exception as e:
            error_str = str(e).lower()
            transient_keywords = (
                "timeout",
                "network",
                "connection",
                "503",
                "502",
                "504",
                "429",
                "rate limit",
                "rate_limit",
            )
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during Kimi extraction: {e}") from e
            raise ProviderPermanentError(f"Error during Kimi extraction: {e}") from e

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        extract_cost_usd = self._estimate_extract_cost_usd(raw_output["usage"])
        cost_usd = extract_cost_usd
        num_pages = page_count(file_path)

        raw_output["extract_cost_usd"] = extract_cost_usd
        raw_output["cost_usd"] = cost_usd
        raw_output["cost_exceeded_budget"] = self._max_cost_usd is not None and cost_usd > self._max_cost_usd
        raw_output["num_pages"] = num_pages
        raw_output["cost_per_page_usd"] = cost_usd / num_pages if num_pages > 0 else None

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=latency_ms,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        return normalize_extract_result(raw_result)
