"""vLLM vision-model one-shot extraction provider.

Talks to an OpenAI-compatible vLLM server that hosts a vision-language model
(e.g. the Qwen3.6-35B-A3B-FP8 self-hosted deployment). The document is rasterized to
one image per page and sent directly to the model — no upstream parse stage —
mirroring the ``*_oneshot_structured_output_file`` cloud-API pipelines but for a
self-hosted vLLM endpoint.

Structured output is requested via the OpenAI ``response_format`` json_schema
form, which vLLM turns into guided decoding (xgrammar). The extract schema is
also inlined into the prompt so the model sees the field names/descriptions.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
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


def _scan_json_structure(s: str) -> tuple[bool, list[str]]:
    """Return (inside_string, stack_of_expected_closers) for a JSON prefix."""
    in_str, stack, _ = _scan_json_structure_detail(s)
    return in_str, stack


def _scan_json_structure_detail(s: str) -> tuple[bool, list[str], int]:
    """As ``_scan_json_structure``, plus where the trailing open string began.

    The third element is the index of the opening quote of the string still
    unterminated at the end of ``s`` (-1 when the prefix does not end inside a
    string). Cutting there discards the half-written value in one step, instead
    of trimming back one comma at a time through a string that may contain
    thousands of them.
    """
    stack: list[str] = []
    in_str = False
    esc = False
    str_start = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                str_start = -1
            continue
        if ch == '"':
            in_str = True
            str_start = i
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack:
                stack.pop()
    return in_str, stack, str_start


def salvage_truncated_json(text: str, max_trims: int = 512) -> Any | None:
    """Best-effort parse of possibly-truncated JSON.

    A vision model that hits ``max_tokens`` returns syntactically incomplete
    JSON (``finish_reason == "length"``). Rather than drop the whole document,
    close the still-open structures — and, if the tail is a half-written
    element, trim back to the last complete one — so the fields that did finish
    still score. Returns ``None`` only when nothing parseable can be recovered.
    """
    s = text.strip()
    if not s:
        return None
    # Strip a leading ```json fence and trailing ``` if present.
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1 :] if nl != -1 else s[3:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    cur = s
    for _ in range(max_trims):
        cur = cur.rstrip()
        if not cur:
            return None
        in_str, _ = _scan_json_structure(cur)
        c = cur + ('"' if in_str else "")
        c = c.rstrip()
        # Drop a trailing comma or a dangling ``"key":`` (value never arrived).
        if c.endswith(","):
            c = c[:-1].rstrip()
        if c.endswith(":"):
            pos = max(c.rfind(","), c.rfind("{"))
            c = c[: pos + 1] if pos != -1 and c[pos] == "{" else (c[:pos] if pos != -1 else c)
            c = c.rstrip()
        # If dropping the dangling key emptied the last element (now a bare
        # opener) and it was a later array/object element (preceded by a comma),
        # drop that element rather than emit a phantom {} / [].
        if c.endswith("{") or c.endswith("["):
            head = c[:-1].rstrip()
            if head.endswith(","):
                c = head[:-1].rstrip()
        # Recompute the closer stack from the ADJUSTED string so it always matches.
        _, stack = _scan_json_structure(c)
        try:
            return json.loads(c + "".join(reversed(stack)))
        except json.JSONDecodeError:
            body = cur.rstrip().rstrip(",")
            # If the tail is a half-written string, drop it whole. Nibbling back
            # comma by comma through a long truncated string burns the trim
            # budget and gives up on output that is otherwise recoverable.
            in_str_now, _, str_start = _scan_json_structure_detail(body)
            if in_str_now and str_start > 0:
                cur = body[:str_start]
                continue
            comma = body.rfind(",")
            opener = max(body.rfind("{"), body.rfind("["))
            if comma > opener:
                cur = body[:comma]
            elif opener != -1:
                cur = body[: opener + 1]
            else:
                return None
    return None


def prune_excess_depth(value: Any, max_depth: int = 100) -> tuple[Any, int]:
    """Replace containers nested deeper than ``max_depth`` with ``None``.

    A degenerate decode (e.g. a bracket loop) can produce parseable JSON nested
    hundreds of levels deep; pydantic-core's serializer then fails the whole
    document with "Circular reference detected (depth exceeded)" at persist
    time. Real extract schemas are at most a dozen levels deep, so pruning far
    below the serializer's guard loses nothing genuine. Returns the pruned
    value and the number of pruned subtrees.
    """
    pruned = 0

    def rec(v: Any, d: int) -> Any:
        nonlocal pruned
        if isinstance(v, (dict, list)) and d >= max_depth:
            pruned += 1
            return None
        if isinstance(v, dict):
            return {k: rec(x, d + 1) for k, x in v.items()}
        if isinstance(v, list):
            return [rec(x, d + 1) for x in v]
        return v

    return rec(value, 0), pruned


@register_provider("vllm_extract")
class VLLMExtractProvider(Provider):
    """One-shot structured extraction through an OpenAI-compatible vLLM VLM.

    Configuration options:
        - server_url (str, required): vLLM server base URL (self-hosted deployment).
        - model (str, required): Served model name (e.g. "qwen3.6-35b-a3b-fp8").
        - api_key_env (str, default "VLLM_API_KEY"): env var for the bearer key.
        - dpi (int, default 150): rasterization DPI for PDF pages.
        - max_pages (int | None, default None): cap on pages sent (None = all).
        - max_tokens (int, default 32768): output-token ceiling.
        - temperature (float, default 0.0): sampling temperature.
        - structured_output (bool, default True): use response_format json_schema
          guided decoding; when False fall back to json_object mode.
        - additional_properties_false (bool, default True): close every object.
        - schema_name (str, default "extraction"): json_schema name field.
        - strict (bool, default False): json_schema strict flag.
        - timeout_s (float, default 900): request timeout.
        - input_price_per_1m / output_price_per_1m (float, default 0.0):
          self-hosted, so cost defaults to 0; override to attribute compute.
        - max_cost_usd (float | None): optional per-doc budget flag.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        # There is no default endpoint: the deployment is yours. A pipeline can
        # name the env var that carries its URL (`endpoint_env_var`) so several
        # self-hosted models can share this provider without hardcoding hosts.
        server_url = self.base_config.get("server_url")
        endpoint_env_var = self.base_config.get("endpoint_env_var")
        if not server_url and endpoint_env_var:
            server_url = os.environ.get(str(endpoint_env_var), "")
        if not server_url:
            raise ProviderConfigError(
                "vllm_extract requires 'server_url' in config"
                + (f" or the {endpoint_env_var} environment variable." if endpoint_env_var else ".")
            )
        self._server_url: str = str(server_url).rstrip("/")

        self._model: str = self.base_config.get("model", "")
        if not self._model:
            raise ProviderConfigError("vllm_extract requires 'model' in config.")

        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        # vLLM tolerates any bearer when --api-key is unset; use a dummy fallback.
        self._api_key: str = os.environ.get(api_key_env, "") or "dummy"

        self._dpi: int = int(self.base_config.get("dpi", 150))
        max_pages_cfg = self.base_config.get("max_pages")
        self._max_pages: int | None = int(max_pages_cfg) if max_pages_cfg is not None else None
        self._max_tokens: int = int(self.base_config.get("max_tokens", 32768))
        self._temperature: float = float(self.base_config.get("temperature", 0.0))
        self._structured_output: bool = bool(self.base_config.get("structured_output", True))
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._schema_name: str = self.base_config.get("schema_name", "extraction")
        self._strict: bool = bool(self.base_config.get("strict", False))
        self._system_prompt: str = self.base_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._user_instruction: str = self.base_config.get("user_instruction", DEFAULT_USER_INSTRUCTION)
        self._timeout_s: float = float(self.base_config.get("timeout_s", self.base_config.get("timeout", 900.0)))
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd")

        self._input_price_per_1m: float = float(self.base_config.get("input_price_per_1m", 0.0))
        self._output_price_per_1m: float = float(self.base_config.get("output_price_per_1m", 0.0))

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=f"{self._server_url}/v1",
            timeout=self._timeout_s,
            # A timed-out generation continues on the remote vLLM server.
            # SDK-level retries create duplicate orphaned generations that
            # consume every GPU slot, so leave retry policy to the benchmark.
            max_retries=0,
        )

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
                f"vllm_extract supports PDFs and {set(IMAGE_EXTENSIONS)}, got {source_path.suffix}"
            )
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProviderPermanentError("pdf2image is required for vllm_extract.") from e

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

    def _build_user_content(
        self, page_images: list[str], schema: dict[str, Any], compact_schema: bool = False
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in page_images
        ]
        # Compact separators only in the context-recovery path: identical schema,
        # ~30% fewer tokens. The default stays pretty-printed so the prompt
        # matches the other one-shot pipelines byte for byte.
        schema_text = json.dumps(schema, separators=(",", ":")) if compact_schema else json.dumps(schema, indent=2)
        text = (
            "Extract every field from the attached document page image(s) according to the JSON schema below.\n\n"
            "JSON schema:\n"
            f"{schema_text}\n\n"
            "Return only one valid JSON object matching the schema. Do not wrap it in markdown fences.\n\n"
            f"{self._user_instruction}"
        )
        content.append({"type": "text", "text": text})
        return content

    @staticmethod
    def _collect_refs(node: Any, out: set[str]) -> None:
        """Collect every ``#/$defs/<name>`` (or ``definitions``) target in a subtree."""
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                out.add(ref.rsplit("/", 1)[-1])
            for v in node.values():
                VLLMExtractProvider._collect_refs(v, out)
        elif isinstance(node, list):
            for v in node:
                VLLMExtractProvider._collect_refs(v, out)

    @classmethod
    def _gc_defs(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """Drop ``$defs``/``definitions`` entries nothing reachable references.

        Extract schemas keep one shared definitions block, so pruning properties
        alone barely shrinks the payload — the definitions they pointed at stay
        behind. Resolve which are still reachable (transitively) and drop the rest.
        """
        out = dict(schema)
        for key in ("$defs", "definitions"):
            defs = out.get(key)
            if not isinstance(defs, dict) or not defs:
                continue
            body = {k: v for k, v in out.items() if k not in ("$defs", "definitions")}
            reachable: set[str] = set()
            cls._collect_refs(body, reachable)
            frontier = list(reachable)
            while frontier:
                name = frontier.pop()
                target = defs.get(name)
                if target is None:
                    continue
                found: set[str] = set()
                cls._collect_refs(target, found)
                for f in found - reachable:
                    reachable.add(f)
                    frontier.append(f)
            out[key] = {k: v for k, v in defs.items() if k in reachable}
        return out

    @staticmethod
    def _prune_schema(schema: dict[str, Any]) -> dict[str, Any] | None:
        """Drop the trailing half of the schema's top-level properties.

        Some ground-truth schemas (full-coverage tax forms) serialize to more
        tokens than the whole context window, so the request can never fit no
        matter how the images are handled. Asking for fewer fields is the same
        bargain we already make with pages and output tokens: a partial
        extraction scores what it found instead of failing the document.
        Returns None when there is nothing left to prune.
        """
        props = schema.get("properties")
        if not isinstance(props, dict) or len(props) <= 1:
            return None
        keys = list(props)
        keep = keys[: max(1, len(keys) // 2)]
        pruned = dict(schema)
        pruned["properties"] = {k: props[k] for k in keep}
        required = schema.get("required")
        if isinstance(required, list):
            pruned["required"] = [r for r in required if r in keep]
        # Dropping properties orphans the definitions they referenced; without
        # this the payload barely shrinks.
        return VLLMExtractProvider._gc_defs(pruned)

    def _response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        if not self._structured_output:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema_name,
                "schema": schema,
                "strict": self._strict,
            },
        }

    def _call_api(
        self,
        schema: dict[str, Any],
        page_images: list[str],
        max_tokens: int | None = None,
        compact_schema: bool = False,
    ) -> dict[str, Any]:
        # Build the request as a dict[str, Any] and splat it in: the OpenAI SDK's
        # typed create() overloads reject our list/dict message+response_format
        # shapes under strict mypy, and passing **kwargs sidesteps that (same
        # approach as deepseek_extract / glm_extract).
        kwargs: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._build_user_content(page_images, schema, compact_schema)},
            ],
            "response_format": self._response_format(schema),
        }
        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        content = getattr(choice.message, "content", "") or ""

        truncated = False
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # A model that hits max_tokens (finish_reason == "length") returns
            # syntactically incomplete JSON. Salvage the completed fields — a
            # truncated extraction earns partial credit, which beats failing the
            # whole document. (Some models also emit stray prose around the JSON;
            # the salvage tolerates a trailing fence/half-element.)
            data = salvage_truncated_json(content)
            if data is None:
                raise ProviderPermanentError(
                    f"vLLM model returned unparseable output (finish_reason={finish_reason}, chars={len(content)}): {e}"
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
                "provider": "vllm_extract",
                "server_url": self._server_url,
                "dpi": self._dpi,
                "max_pages": self._max_pages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "structured_output": self._structured_output,
                "additional_properties_false": self._additional_properties_false,
                "strict": self._strict,
                "max_cost_usd": self._max_cost_usd,
                "input_price_per_1m": self._input_price_per_1m,
                "output_price_per_1m": self._output_price_per_1m,
            },
        }

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
        output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
        total_tokens = getattr(usage, "total_tokens", None) or input_tokens + output_tokens
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }

    # Substrings that identify a vLLM context-length rejection (400). The exact
    # wording varies ("maximum context length is N tokens ... reduce the
    # length", "longer than the maximum model length", etc.).
    _CONTEXT_ERROR_MARKERS = (
        "maximum context length",
        "maximum model length",
        "context length",
        "context window",
        "longer than the maximum",
        "reduce the length",
        "please reduce",
        "too many tokens",
    )

    @classmethod
    def _is_context_length_error(cls, exc: Exception) -> bool:
        m = str(exc).lower()
        return any(k in m for k in cls._CONTEXT_ERROR_MARKERS)

    # vLLM's context-rejection message carries the exact budget numbers, e.g.
    # "This model's maximum context length is 40960 tokens. However, you
    #  requested 32768 output tokens and your prompt contains at least 8193
    #  input tokens, for a total of at least 40961 tokens."
    # Two shapes are seen in the wild and both must parse, since falling through
    # sends the request down the lossy image/page path when all it needed was a
    # smaller output budget:
    #   "...maximum context length is 40960 tokens. However, you requested 32768
    #    output tokens and your prompt contains at least 8193 input tokens..."
    #   "'max_tokens' ... is too large: 32768. This model's maximum context
    #    length is 40960 tokens and your request has 9519 input tokens..."
    _CTX_LIMIT_RE = re.compile(r"maximum (?:context|model) length is (\d+)", re.IGNORECASE)
    _CTX_INPUT_RE = re.compile(r"(?:at least|has|contains|have)\s+(\d+)\s+(?:input|prompt) tokens", re.IGNORECASE)

    @classmethod
    def _parse_context_numbers(cls, message: str) -> tuple[int, int] | None:
        """Return (context_limit, input_tokens) from a vLLM context rejection."""
        limit = cls._CTX_LIMIT_RE.search(message)
        inp = cls._CTX_INPUT_RE.search(message)
        if not limit or not inp:
            return None
        return int(limit.group(1)), int(inp.group(1))

    # A dense scanned page can leave under 1k tokens of headroom, so the floor
    # has to be small enough that "a little output" still beats failing.
    _MIN_OUTPUT_TOKENS = 256
    _CTX_SAFETY_MARGIN = 256
    # Image tokens dominate the input, so shrinking the rasterized pages is the
    # most effective way to reclaim context — and unlike dropping pages it keeps
    # every page of the document visible to the model.
    _DOWNSCALE_FACTOR = 0.75
    _MAX_DOWNSCALE_ROUNDS = 4
    # Above this page count, dropping trailing pages is cheaper than resizing
    # every page (and is the path already validated on the 192-page documents).
    _DROP_BEFORE_SCALE_PAGES = 16
    # ~25k tokens of schema text: past this the schema, not the pages, is what
    # blows the context, so shrink it before touching the images.
    _HUGE_SCHEMA_CHARS = 100_000
    _MAX_SCHEMA_PRUNE_ROUNDS = 12

    @classmethod
    def _downscale_page_images(cls, page_images: list[str]) -> list[str]:
        """Return the pages re-encoded at ``_DOWNSCALE_FACTOR`` linear scale."""
        from PIL import Image

        out: list[str] = []
        for b64 in page_images:
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            w, h = img.size
            new_size = (max(64, int(w * cls._DOWNSCALE_FACTOR)), max(64, int(h * cls._DOWNSCALE_FACTOR)))
            resized = img.resize(new_size)
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return out

    def _call_api_fitting_context(self, schema: dict[str, Any], page_images: list[str]) -> dict[str, Any]:
        """Call the model, shrinking the request until it fits the model context.

        A request can overflow the served context window two ways: the page
        images + schema exceed it outright (very long documents), or the fixed
        output budget leaves too little input headroom on a small-context model.
        vLLM rejects both with a 400 before generating. Rather than fail the
        document, first shrink ``max_tokens`` to what the rejection says fits
        (output truncation earns partial credit), and drop trailing pages when
        the input itself is too big — truncation is acceptable, a context-length
        error is not.
        """
        pages = page_images
        dropped = 0
        scale_rounds = 0
        prune_rounds = 0
        cur_schema = schema
        compact = False
        out_budget = self._max_tokens
        while True:
            try:
                out = self._call_api(cur_schema, pages, max_tokens=out_budget, compact_schema=compact)
                if dropped:
                    out["pages_truncated"] = dropped
                    out["pages_sent"] = len(pages)
                    out["truncated"] = True
                if out_budget < self._max_tokens:
                    out["max_tokens_fitted"] = out_budget
                if scale_rounds:
                    out["image_downscale_rounds"] = scale_rounds
                    out["truncated"] = True
                if prune_rounds:
                    out["schema_pruned_rounds"] = prune_rounds
                    out["schema_fields_sent"] = len(cur_schema.get("properties") or {})
                    out["truncated"] = True
                return out
            except Exception as e:
                if not self._is_context_length_error(e):
                    raise
                # When the schema text alone rivals the context window (some
                # full-coverage form schemas serialize to >100k tokens), nothing
                # done to the images can ever make the request fit — shrink the
                # schema first instead of burning rounds on the pages.
                if not compact and len(json.dumps(cur_schema)) > self._HUGE_SCHEMA_CHARS:
                    compact = True
                    continue
                if compact and prune_rounds < self._MAX_SCHEMA_PRUNE_ROUNDS:
                    smaller = self._prune_schema(cur_schema)
                    if smaller is not None:
                        cur_schema = smaller
                        prune_rounds += 1
                        continue
                # Preferred: compute the output budget that fits from the
                # server's own numbers and retry with every page intact.
                nums = self._parse_context_numbers(str(e))
                if nums:
                    limit, inp = nums
                    fit = limit - inp - self._CTX_SAFETY_MARGIN
                    if self._MIN_OUTPUT_TOKENS <= fit < out_budget:
                        out_budget = fit
                        continue
                # The input itself is too big. For a long document, dropping
                # trailing pages is the cheapest lever; for a short one (a dense
                # scanned form is only a few huge pages), shrink the images
                # instead so no page is lost.
                if len(pages) > self._DROP_BEFORE_SCALE_PAGES:
                    keep = max(1, (len(pages) * 3) // 4)
                    dropped += len(pages) - keep
                    pages = pages[:keep]
                    continue
                if scale_rounds < self._MAX_DOWNSCALE_ROUNDS:
                    pages = self._downscale_page_images(pages)
                    scale_rounds += 1
                    continue
                if len(pages) > 1:
                    keep = max(1, (len(pages) * 3) // 4)
                    keep = min(keep, len(pages) - 1)
                    dropped += len(pages) - keep
                    pages = pages[:keep]
                    continue
                # One small page and still no room: halve the output budget
                # down to the floor before giving up.
                if out_budget > self._MIN_OUTPUT_TOKENS:
                    out_budget = max(self._MIN_OUTPUT_TOKENS, out_budget // 2)
                    continue
                raise

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
            raw_output = self._call_api_fitting_context(schema, page_images)
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
                # A self-hosted vLLM web endpoint can drop a request in flight
                # (observed as HTTP 408 with body "Missing request, possibly due
                # to expiry or cancellation"); an immediate retry succeeds.
                "408",
                "missing request",
                # A vLLM engine crash (e.g. CUDA OOM under a heavy multi-image
                # batch) 500s every in-flight request at once; the container
                # restarts and an immediate retry succeeds, so treat these as
                # transient rather than failing the documents.
                "enginecore",
                "internal server error",
                "internalservererror",
            )
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during vLLM extraction: {e}") from e
            raise ProviderPermanentError(f"Error during vLLM extraction: {e}") from e

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        usage = raw_output["usage"]
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        extract_cost_usd = (
            input_tokens / 1_000_000 * self._input_price_per_1m + output_tokens / 1_000_000 * self._output_price_per_1m
        )
        cost_usd = extract_cost_usd
        num_pages = page_count(file_path)

        raw_output["extract_cost_usd"] = extract_cost_usd
        raw_output["cost_usd"] = cost_usd
        raw_output["cost_exceeded_budget"] = self._max_cost_usd is not None and cost_usd > self._max_cost_usd
        raw_output["num_pages"] = num_pages
        if num_pages > 0:
            raw_output["cost_per_page_usd"] = cost_usd / num_pages

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
