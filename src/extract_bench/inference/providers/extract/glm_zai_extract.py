"""GLM (z.ai) direct-file structured extraction provider.

GLM-5.3-flash extraction over the raw document, through z.ai's
OpenAI-compatible chat completions endpoint. The source file is sent inline —
PDFs (and other document types z.ai accepts: txt, docx, xlsx, pptx) as a
``file_url`` block, raw images as an ``image_url`` block — both base64 data URLs;
z.ai has no Files API, so nothing is uploaded first. Structured output is
requested via ``response_format={"type": "json_object"}`` with the target JSON
schema embedded in the prompt (the same approach as the parsed-text
``glm_extract`` provider, which is proven against GLM chat completions).

Thinking is always on for GLM-5.3-flash; ``reasoning_tokens`` are part of
``completion_tokens`` and billed at the output rate.
"""

from __future__ import annotations

import base64
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
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType

# z.ai list pricing: USD per million tokens (input, cached_input, output).
# A 50%-off promo (0.075 / 0.015 / 0.25) runs through 2026-09-09; list price is
# used so the benchmark cost stays stable after it ends.
# Source: https://docs.z.ai/guides/overview/pricing (verified 2026-08-26)
_GLM_ZAI_EXTRACT_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    "glm-5.3-flash": (0.15, 0.03, 0.50),
}

_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"


@register_provider("glm_zai_extract")
class GLMZaiExtractProvider(Provider):
    """Structured extraction over the raw document via GLM-5.3-flash on z.ai."""

    DEFAULT_MODEL = "glm-5.3-flash"
    # Use the model's full output ceiling. Thinking is always on and shares the
    # max_tokens budget with the visible JSON, so long-document / long-list
    # schemas need the largest window available -- a max_tokens stop is a hard
    # error here, not a silent truncation. z.ai caps max_tokens at 131072 for
    # glm-5.3-flash (requests above that return a 400 stating the [1, 131072]
    # range; verified 2026-08-26).
    DEFAULT_MAX_TOKENS = 131072

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._api_key = self.base_config.get("api_key") or os.getenv("GLM_ZAI_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "GLM z.ai API key is required. Set GLM_ZAI_API_KEY or pass api_key in base_config."
            )

        self._model: str = self.base_config.get("model", self.DEFAULT_MODEL)
        self._base_url: str = self.base_config.get("base_url", _ZAI_BASE_URL)
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._system_prompt: str = self.base_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._user_instruction: str = self.base_config.get("user_instruction", DEFAULT_USER_INSTRUCTION)
        self._temperature: float = float(self.base_config.get("temperature", 0))
        self._reasoning_effort: str | None = self.base_config.get("reasoning_effort")
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd")
        self._api_timeout_s: float = float(
            self.base_config.get("api_timeout_s", self.base_config.get("timeout_s", 900.0))
        )
        max_tokens_cfg = self.base_config.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        self._max_tokens: int | None = int(max_tokens_cfg) if max_tokens_cfg is not None else None

        # Optional pipeline-level schema override kept inline so provider
        # configurations remain portable.
        config_schema = self.base_config.get("schema_override")
        self._schema_override_from_config: dict[str, Any] | None = (
            config_schema if isinstance(config_schema, dict) else None
        )

        default_input, default_cached_input, default_output = self._pricing_for_model(self._model)
        self._input_price_per_1m: float = float(self.base_config.get("input_price_per_1m", default_input))
        self._cached_input_price_per_1m: float = float(
            self.base_config.get("cached_input_price_per_1m", default_cached_input)
        )
        self._output_price_per_1m: float = float(self.base_config.get("output_price_per_1m", default_output))

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._api_timeout_s)

    @staticmethod
    def _pricing_for_model(model: str) -> tuple[float, float, float]:
        matches = [
            (prefix, rates) for prefix, rates in _GLM_ZAI_EXTRACT_PRICING_PER_M.items() if model.startswith(prefix)
        ]
        return max(matches, key=lambda item: len(item[0]))[1] if matches else (0.0, 0.0, 0.0)

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        schema = promote_repeated_structure(schema)
        if self._additional_properties_false:
            return add_additional_properties_false(schema)
        return schema

    def _build_file_block(self, source_path: Path) -> dict[str, Any]:
        ext = source_path.suffix.lower()
        with open(source_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        if ext in IMAGE_EXTENSIONS:
            mime = IMAGE_EXTENSIONS[ext]
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        # PDFs and other document types z.ai accepts (txt, docx, xlsx, pptx)
        # ride in a file_url block.
        return {"type": "file_url", "file_url": {"url": f"data:application/pdf;base64,{b64}"}}

    def _build_prompt(self, schema: dict[str, Any]) -> str:
        return (
            "Extract every field from the attached document according to the JSON schema.\n\n"
            "JSON schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Return only one valid JSON object. Do not wrap it in markdown fences.\n\n"
            f"{self._user_instruction}"
        )

    def _call_api(self, schema: dict[str, Any], source_path: Path) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        self._build_file_block(source_path),
                        {"type": "text", "text": self._build_prompt(schema)},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise ProviderPermanentError("GLM hit max_tokens before completing the JSON response. Increase max_tokens.")

        content = getattr(choice.message, "content", "") or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ProviderPermanentError(f"GLM returned non-JSON output despite response_format: {e}") from e

        usage = self._extract_usage(response)
        return {
            "data": data,
            "model": self._model,
            "usage": usage,
            "_config": {
                "provider": "glm_zai",
                "base_url": self._base_url,
                "input_mode": "file",
                "additional_properties_false": self._additional_properties_false,
                "temperature": self._temperature,
                "reasoning_effort": self._reasoning_effort,
                "max_tokens": self._max_tokens,
                "api_timeout_s": self._api_timeout_s,
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
        cached_input_tokens = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached_input_tokens = getattr(details, "cached_tokens", None) or 0

        return {
            "input_tokens": int(input_tokens),
            "cached_input_tokens": int(cached_input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }

    def _estimate_cost_usd(self, usage: dict[str, int]) -> float:
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
            "pricing_basis": "glm_5_3_flash_zai_list",
            "input_price_per_1m": self._input_price_per_1m,
            "cached_input_price_per_1m": self._cached_input_price_per_1m,
            "output_price_per_1m": self._output_price_per_1m,
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"{type(self).__name__} only supports EXTRACT product type, got {request.product_type}"
            )
        effective_schema = self._schema_override_from_config or request.schema_override
        if not effective_schema:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override or via the schema_override pipeline config."
            )

        started_at = datetime.now()
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        schema = self._prepare_schema(effective_schema)

        try:
            raw_output = self._call_api(schema, file_path)
        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
            raise
        except Exception as e:
            error_str = str(e).lower()
            transient_keywords = ("timeout", "network", "connection", "503", "502", "504", "429", "rate limit", "500")
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during GLM extraction: {e}") from e
            raise ProviderPermanentError(f"Error during GLM extraction: {e}") from e

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        num_pages = page_count(file_path)
        cost_usd = self._estimate_cost_usd(raw_output["usage"])
        raw_output["extract_cost_usd"] = cost_usd
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
