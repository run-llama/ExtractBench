"""Provider for NuExtract3 structured extraction on the self-hosted vLLM server.

NuExtract3 (numind/NuExtract3, a 4B Qwen3.5-based VLM) does structured
extraction natively: given document images + a NuExtract *template* it emits a
JSON object shaped like that template. It reuses the same deployed vLLM endpoint
as the parse pipeline — extraction is just a different ``chat_template_kwargs``
(a ``template`` instead of ``mode="markdown"``).

The bench supplies a JSON Schema (``request.schema_override``). NuExtract expects
its own template format whose leaves are type names (``"string"``, ``"integer"``,
``"number"``, ``"boolean"``, ``"date"``…) and whose enums are lists of options,
so we convert the JSON Schema to a NuExtract template first with numind's
official ``convert_json_schema_to_nuextract_template`` utility.

Like lift, NuExtract3 emits schema-shaped JSON with no per-field citations /
bboxes, so ``field_citations`` is always empty (evidence-bbox metrics are N/A);
the bench scores the extracted values.
"""

import asyncio
import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

try:
    from numind.nuextract_utils import convert_json_schema_to_nuextract_template
except ImportError:  # Optional runner dependency; checked when the provider runs.
    convert_json_schema_to_nuextract_template = None  # type: ignore[assignment]

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

DEFAULT_SERVED_MODEL_NAME = "nuextract3"

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _repair_truncated_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of a truncated / trailing-garbage JSON object.

    Large extractions (e.g. a 13F with hundreds of holdings) can exceed the
    output-token budget and get cut mid-array, and models occasionally append
    junk after a valid object. We recover the top-level fields plus every
    complete array element before the cut by truncating at successive value/
    container boundaries (latest first), closing any still-open brackets, and
    returning the first prefix that parses to an object. Returns ``None`` if
    nothing parseable can be salvaged.
    """
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]

    # Collect candidate cut points: the index just after every completed string
    # or closed container — i.e. positions we can truncate at and then re-close.
    boundaries: list[int] = []
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                boundaries.append(i + 1)
            continue
        if ch == '"':
            in_str = True
        elif ch in "}]":
            boundaries.append(i + 1)

    # Try the latest boundaries first; the valid cut is usually within a handful
    # of the truncation point. Cap attempts so a pathological output stays cheap.
    for cut in list(reversed(boundaries))[:300]:
        prefix = s[:cut]
        stack: list[str] = []
        p_in_str = False
        p_esc = False
        for ch in prefix:
            if p_in_str:
                if p_esc:
                    p_esc = False
                elif ch == "\\":
                    p_esc = True
                elif ch == '"':
                    p_in_str = False
                continue
            if ch == '"':
                p_in_str = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]" and stack:
                stack.pop()
        candidate = re.sub(r"[,\s]+$", "", prefix) + "".join(reversed(stack))
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None



def _json_schema_to_nuextract_template_and_instructions(node: Any) -> tuple[Any, str]:
    """Convert a JSON Schema and render the converter's descriptions as instructions."""
    if not isinstance(node, dict):
        raise ProviderPermanentError("JSON Schema must be an object")
    if convert_json_schema_to_nuextract_template is None:
        raise ProviderConfigError(
            "The numind SDK is required for NuExtract template conversion. "
            "Install the 'runners' extra or run `pip install numind`."
        )

    try:
        processed_schema = convert_json_schema_to_nuextract_template(node)
        template = processed_schema["template"]
        descriptions = processed_schema["descriptions"]
    except (KeyError, TypeError, ValueError) as e:
        raise ProviderPermanentError(f"Could not convert JSON Schema to a NuExtract template: {e}") from e
    return template, "\n".join(descriptions)


@register_provider("nuextract3_extract")
class NuExtract3ExtractProvider(Provider):
    """
    Provider for NuExtract3 structured extraction (direct vLLM, self-hosted).

    Configuration options:
        - server_url (str, required): self-hosted vLLM server URL (the same endpoint
          the nuextract3 parse pipeline uses).
        - model (str, default="nuextract3"): served model name.
        - timeout (int, default=1800): per-request timeout in seconds (large
          extractions generate a lot of tokens and can run for many minutes).
        - dpi (int, default=150): DPI for PDF-to-image rendering.
        - max_pages (int, default=100): cap on rendered pages per document (keeps
          the request within the server's image / context limits).
        - max_tokens (int, default=100000): max output tokens — large enough for
          a long holdings array (server context is 262144).
        - temperature (float, default=0.2): sampling temperature (non-thinking).
        - enable_thinking (bool, default=False): NuExtract reasoning mode.
        - api_key_env (str, default="VLLM_API_KEY"): env var for the API key.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        # No default endpoint: the deployment is yours. `endpoint_env_var` lets
        # the pipeline name the env var that carries its URL.
        server_url = self.base_config.get("server_url")
        endpoint_env_var = self.base_config.get("endpoint_env_var")
        if not server_url and endpoint_env_var:
            import os as _os

            server_url = _os.environ.get(str(endpoint_env_var), "")
        if not server_url:
            raise ProviderConfigError(
                "nuextract3_extract provider requires 'server_url' in config"
                + (f" or the {endpoint_env_var} environment variable." if endpoint_env_var else ".")
            )
        self._server_url: str = str(server_url)

        self._model = self.base_config.get("model", DEFAULT_SERVED_MODEL_NAME)
        self._timeout = int(self.base_config.get("timeout", 1800))
        self._dpi = int(self.base_config.get("dpi", 150))
        self._max_pages = int(self.base_config.get("max_pages", 100))
        self._max_tokens = int(self.base_config.get("max_tokens", 100000))
        self._temperature = float(self.base_config.get("temperature", 0.2))
        self._enable_thinking = bool(self.base_config.get("enable_thinking", False))

        import os

        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")

    # ------------------------------------------------------------------
    # Image rendering
    # ------------------------------------------------------------------

    def _render_images_b64(self, file_path: Path) -> list[str]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pdf2image import convert_from_path

                images = convert_from_path(file_path, dpi=self._dpi)
            except ImportError as e:
                raise ProviderPermanentError("pdf2image is required.") from e
            except Exception as e:
                raise ProviderPermanentError(f"Error converting PDF to image: {e}") from e
            if not images:
                raise ProviderPermanentError(f"No pages found in PDF: {file_path}")
            images = images[: self._max_pages]
            out: list[str] = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                out.append(base64.b64encode(buf.getvalue()).decode())
            return out

        if suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            return [base64.b64encode(file_path.read_bytes()).decode()]

        raise ProviderPermanentError(
            f"Unsupported file type: {suffix}. Supported: .pdf, .png, .jpg, .jpeg, .webp, .tiff, .bmp"
        )

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    async def _extract_async(
        self,
        images_b64: list[str],
        template: dict[str, Any],
        instructions: str,
    ) -> dict[str, Any]:
        api_url = f"{self._server_url.rstrip('/')}/v1/chat/completions"

        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in images_b64]
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
            # NuExtract selects structured-extraction by passing a template
            # (vLLM OpenAI extension — top-level request field).
            "chat_template_kwargs": {
                "template": json.dumps(template),
                "instructions": instructions,
                "enable_thinking": self._enable_thinking,
            },
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    if resp.status in (408, 429, 502, 503, 504):
                        raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                    raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

                result: dict[str, Any] = await resp.json()

        try:
            raw_content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderPermanentError(f"Invalid response format: {e}") from e
        if not raw_content:
            raise ProviderPermanentError("Empty content response from API")

        return {
            "content": str(raw_content),
            "template": template,
            "instructions": instructions,
            "_config": {
                "server_url": self._server_url,
                "model": self._model,
                "dpi": self._dpi,
            },
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(f"NuExtract3ExtractProvider only supports EXTRACT, got {request.product_type}")
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override"
            )

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        template, instructions = _json_schema_to_nuextract_template_and_instructions(request.schema_override)
        if not isinstance(template, dict):
            raise ProviderPermanentError("Top-level schema must be an object (produced a non-dict template)")

        try:
            images_b64 = self._render_images_b64(file_path)
            raw_output = asyncio.run(self._extract_async(images_b64, template, instructions))

            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

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

        except (ProviderPermanentError, ProviderTransientError):
            raise
        except TimeoutError as e:
            raise ProviderTransientError(f"Request timed out after {self._timeout}s") from e
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_extraction(content: str) -> dict[str, Any]:
        """Parse the model's JSON extraction, never raising.

        Tries a strict parse (raw + de-fenced), then best-effort salvage of a
        truncated/garbage object. Returns ``{}`` if nothing usable is found, so
        a single pathological document (truncated huge array, repetition loop)
        is scored on whatever it produced rather than hard-failing the whole run.
        """
        text = _THINK_RE.sub("", content).strip()
        candidates = [text]
        fence = _FENCE_RE.search(text)
        if fence:
            candidates.append(fence.group(1))

        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed

        # Salvage: recover top-level fields + complete array elements before a cut.
        for cand in candidates:
            repaired = _repair_truncated_json_object(cand)
            if repaired is not None:
                return repaired
        return {}

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"NuExtract3ExtractProvider only supports EXTRACT, got {raw_result.product_type}"
            )

        content = raw_result.raw_output.get("content", "")
        extracted_data = self._parse_extraction(content) if content else {}

        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data,
            field_citations=[],  # NuExtract3 emits no per-field citations / bboxes
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
