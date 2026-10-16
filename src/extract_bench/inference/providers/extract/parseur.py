"""Provider for Parseur (https://parseur.com) schema-guided extraction.

Parseur's REST API (https://developer.parseur.com) takes no JSON Schema: a
mailbox carries a field tree of scalar and ``TABLE`` fields. The provider
converts each schema into such a tree (:func:`convert_json_schema`), keeps one
mailbox per converted schema, uploads the PDF, polls the document and maps the
result back to the schema's shape (:func:`decode_result`).

Environment: ``PARSEUR_API_KEY`` (required). Parseur returns no per-field
evidence, so ``field_citations`` stays empty.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.direct_model_utils import page_count
from extract_bench.inference.providers.extract.table_codegen.schema_utils import resolve_refs
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.parseur.com"
DEFAULT_AI_ENGINE = "GCP_AI_2"  # AI Vision engine v3
MAILBOX_PREFIX = "ExtractBench "
HTTP_TIMEOUT_S = 120.0
# The upload response comes once the server has stored the file.
UPLOAD_TIMEOUT_S = 600.0
# The runner runs 20 documents at once by default; requests' default pool keeps only 10 connections.
HTTP_POOL_SIZE = 32
# Consecutive polling errors (network, 5xx) tolerated before giving up on a document. A 429 does not
# count: the polling deadline bounds it.
MAX_POLL_ERRORS = 10
# A document can read PARSEDOK a moment before its result is available: poll again shortly, a few times.
MAX_EMPTY_RESULT_POLLS = 3
EMPTY_RESULT_POLL_INTERVAL_S = 0.5

# Document statuses that mean "still processing" (mirrors parseur-py).
PENDING_STATUSES = frozenset({"INCOMING", "ANALYZING", "PROGRESS"})
# EXPORTKO: the extraction succeeded and only an export (webhook, integration) of the account failed.
SUCCESS_STATUSES = frozenset({"PARSEDOK", "EXPORTKO"})

SCALAR_ARRAY_COLUMN = "value"
BOOLEAN_CHOICES = ["true", "false"]


# =============================================================================
# JSON Schema -> Parseur field tree
# =============================================================================


def _is_null_schema(node: Any) -> bool:
    return isinstance(node, dict) and node.get("type") == "null"


def _simplify(node: Any) -> dict[str, Any]:
    """Collapse nullable unions to the non-null branch (``$ref`` is already resolved).

    Multi-branch unions of different non-null shapes fall back to a plain
    ``string`` (Parseur ``TEXT``) so the value is at least captured verbatim.
    """
    if not isinstance(node, dict):
        return {"type": "string"}

    type_field = node.get("type")
    if isinstance(type_field, list):
        non_null = [t for t in type_field if t != "null"]
        node = {**node, "type": non_null[0] if len(non_null) == 1 else "string"}
        if len(non_null) != 1:
            node.pop("enum", None)

    for union_key in ("anyOf", "oneOf", "allOf"):
        options = node.get(union_key)
        if not isinstance(options, list) or not options:
            continue
        non_null = [o for o in options if not _is_null_schema(o)]
        base = {k: v for k, v in node.items() if k != union_key}
        if len(non_null) == 1:
            resolved = _simplify(non_null[0])
            # Outer keys (description, title, default) win over the branch's.
            return _simplify({**resolved, **{k: v for k, v in base.items() if k != "default"}})
        if union_key == "allOf" and non_null:
            merged = dict(base)
            for option in non_null:
                merged.update(_simplify(option))
            return _simplify(merged)
        # Several distinct shapes: capture as text.
        fallback = {k: v for k, v in base.items() if k not in ("enum", "format", "properties", "items")}
        fallback["type"] = "string"
        return fallback

    if "type" not in node:
        node = {**node, "type": "object" if "properties" in node else "array" if "items" in node else "string"}
    return node


def _join_description(*parts: Any) -> str | None:
    texts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return " ".join(texts) or None


def _scalar_field(name: str, node: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(parser_object, decode_spec)`` for a scalar schema node."""
    node_type = node.get("type")
    description = node.get("description")
    field: dict[str, Any] = {"name": name}
    spec: dict[str, Any] = {"cast": None}

    enum = node.get("enum")
    if node_type == "boolean":
        field["format"] = "ONELINE"
        field["choice_set"] = list(BOOLEAN_CHOICES)
        spec["cast"] = "bool"
        description = _join_description(description, 'Answer "true" or "false".')
    elif isinstance(enum, list) and enum:
        choices = [str(v) for v in enum if v is not None]
        if choices:
            field["format"] = "ONELINE"
            field["choice_set"] = sorted(set(choices))
            # Restore the original JSON type of enum members (e.g. integers).
            spec["enum_map"] = {str(v): v for v in enum if v is not None}
        else:
            field["format"] = "TEXT"
    elif node_type in ("integer", "number"):
        # Parseur returns a NUMBER field as a JSON number.
        field["format"] = "NUMBER"
    elif node_type == "string" and node.get("format") == "date":
        field["format"] = "DATE"
    else:
        # A JSON Schema string does not say whether line breaks belong to the value: single-line, the app's default.
        field["format"] = "ONELINE"

    if description:
        field["query"] = description
    return field, spec


def convert_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON Schema into a Parseur ``parser_object_set``.

    Nested objects become dotted field names, arrays of objects become tables, arrays of
    scalars become single-column tables, booleans become a ``true``/``false`` choice set.
    Descriptions become field instructions.

    Returns a JSON-serializable dict::

        {
          "parser_object_set": [...],  # payload for POST /parser
          "fields": {name: spec, ...},  # how to decode each top-level field
          "dropped": ["path.to.nested.table", ...],  # schema paths with no Parseur shape
        }

    ``spec`` is ``{"kind": "scalar"|"table"|"scalar_array", "path": [...],
    "cast": ..., "enum_map": {...}, "columns": {col_name: {"path": [...], ...}}}``.
    """
    root = resolve_refs(schema) if isinstance(schema, dict) else {}
    top = _simplify(root)
    if top.get("type") != "object" or not isinstance(top.get("properties"), dict):
        raise ValueError("Only object-rooted JSON Schemas with properties are supported")

    dropped: list[str] = []

    def walk(properties: dict[str, Any], path: list[str], emit_field: Any, emit_table: Any) -> None:
        """Hand each scalar to ``emit_field`` and each array to ``emit_table``, flattening objects."""
        for key, raw_sub in properties.items():
            sub = _simplify(raw_sub)
            sub_path = [*path, key]
            sub_type = sub.get("type")

            if sub_type == "object":
                props = sub.get("properties")
                if not isinstance(props, dict) or not props:
                    dropped.append(".".join(sub_path))
                    continue
                description = sub.get("description")
                if isinstance(description, str) and description.strip():
                    # Parseur has no object field, so the object's own description has no home:
                    # each child carries it in its instructions.
                    prefix = f"[{key}: {description.strip()}]"
                    simplified = {k: _simplify(v) for k, v in props.items()}
                    props = {
                        k: {**v, "description": _join_description(prefix, v.get("description"))}
                        for k, v in simplified.items()
                    }
                walk(props, sub_path, emit_field, emit_table)
            elif sub_type == "array":
                emit_table(sub_path, sub)
            else:
                emit_field(sub_path, sub)

    def build_table(
        path: list[str], node: dict[str, Any], base_len: int
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Build the TABLE field for an array, named ``path[base_len:]``.

        Returns ``(parser_object, decode_spec)``, or ``None`` when the array has no Parseur shape
        and has been recorded as dropped.
        """
        items = _simplify(node.get("items")) if isinstance(node.get("items"), dict) else {"type": "string"}
        query = _join_description(node.get("description"), items.get("description"))
        table: dict[str, Any] = {"name": ".".join(path[base_len:]), "format": "TABLE"}
        if query:
            table["query"] = query

        if items.get("type") == "array":
            dropped.append(".".join(path))
            return None

        if items.get("type") != "object":
            # Array of scalars -> single-column table.
            column, spec = _scalar_field(SCALAR_ARRAY_COLUMN, items)
            if query:
                column.setdefault("query", query)
            table["parser_object_set"] = [column]
            return table, {"kind": "scalar_array", "path": path[base_len:], **spec}

        columns: list[dict[str, Any]] = []
        column_specs: dict[str, Any] = {}

        def emit_column(col_path: list[str], col_node: dict[str, Any]) -> None:
            col_name = ".".join(col_path[len(path) :])
            field, spec = _scalar_field(col_name, col_node)
            columns.append(field)
            column_specs[col_name] = {"path": col_path[len(path) :], **spec}

        def emit_nested_table(col_path: list[str], col_node: dict[str, Any]) -> None:
            built = build_table(col_path, col_node, len(path))
            if built is not None:
                columns.append(built[0])
                column_specs[built[0]["name"]] = built[1]

        props = items.get("properties")
        if isinstance(props, dict):
            walk(props, path, emit_column, emit_nested_table)
        if not columns:
            dropped.append(".".join(path))
            return None
        table["parser_object_set"] = columns
        return table, {"kind": "table", "path": path[base_len:], "columns": column_specs}

    parser_object_set: list[dict[str, Any]] = []
    fields: dict[str, Any] = {}

    def emit_top_field(path: list[str], node: dict[str, Any]) -> None:
        field, spec = _scalar_field(".".join(path), node)
        parser_object_set.append(field)
        fields[field["name"]] = {"kind": "scalar", "path": path, **spec}

    def emit_top_table(path: list[str], node: dict[str, Any]) -> None:
        built = build_table(path, node, 0)
        if built is not None:
            parser_object_set.append(built[0])
            fields[built[0]["name"]] = built[1]

    walk(top["properties"], [], emit_top_field, emit_top_table)
    return {"parser_object_set": parser_object_set, "fields": fields, "dropped": dropped}


def conversion_hash(conversion: dict[str, Any], *, ai_engine: str, ai_instructions: str | None) -> str:
    """Stable id for a converted schema + the mailbox settings that shape extraction.

    12 hex chars: short enough to keep the mailbox name legible in the Parseur app, long
    enough that two schemas never share a mailbox by accident.
    """
    payload = {
        "ai_engine": ai_engine,
        "ai_instructions": ai_instructions or "",
        "parser_object_set": conversion["parser_object_set"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _parse_document_result(document: dict[str, Any]) -> dict[str, Any] | None:
    """The API serializes ``result`` as a JSON string; :meth:`ParseurProvider.run_inference` stores it parsed."""
    result = document.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return None
    return result if isinstance(result, dict) else None


# =============================================================================
# Parseur result -> schema-shaped JSON
# =============================================================================


def _cast_scalar(value: Any, spec: dict[str, Any]) -> Any:
    """Undo the provider's scalar encodings: the boolean choice set and the JSON type of enum members.

    The server passes an empty text value through as "", which the scorer does not count as null.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None

    enum_map = spec.get("enum_map")
    if enum_map:
        return enum_map.get(str(value).strip(), value)

    if spec.get("cast") == "bool":
        return {"true": True, "false": False}.get(str(value).strip().lower())
    return value


_MISSING = object()


def _lookup(container: dict[str, Any], name: str, path: list[str]) -> Any:
    """Find a field by its flat dotted name (table rows keep dotted keys) or by its nested path
    (top-level fields come back nested with ``expand_result``)."""
    if name in container:
        return container[name]
    node: Any = container
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _set_path(target: dict[str, Any], path: list[str], value: Any) -> None:
    node = target
    for key in path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[path[-1]] = value


def _decode_value(raw: Any, spec: dict[str, Any]) -> Any:
    """Decode one field value: a scalar, a table (list of row dicts) or a scalar list (``[{"value": x}]``)."""
    if raw is _MISSING or raw is None:
        return None
    kind = spec.get("kind", "scalar")
    if kind == "scalar":
        return _cast_scalar(raw, spec)
    rows = raw if isinstance(raw, list) else []
    if kind == "scalar_array":
        return [_cast_scalar(row.get(SCALAR_ARRAY_COLUMN) if isinstance(row, dict) else row, spec) for row in rows]
    return [_decode_fields(row, spec["columns"], keep_missing=True) for row in rows if isinstance(row, dict)]


def _decode_fields(container: dict[str, Any], specs: dict[str, Any], *, keep_missing: bool) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for name, spec in specs.items():
        raw = _lookup(container, name, spec["path"])
        if raw is not _MISSING or keep_missing:
            _set_path(decoded, list(spec["path"]), _decode_value(raw, spec))
    return decoded


def decode_result(result: dict[str, Any], conversion: dict[str, Any]) -> dict[str, Any]:
    """Rebuild schema-shaped JSON from a Parseur document ``result``; values are never repaired.

    A table row carries every column the mailbox defines, the ones Parseur returned empty as
    ``None``, so every row has the same shape. A top-level field Parseur did not return is omitted.
    """
    return _decode_fields(result, conversion["fields"], keep_missing=False)


# =============================================================================
# Provider
# =============================================================================


@register_provider("parseur")
class ParseurProvider(Provider):
    """Parseur EXTRACT provider over the public REST API.

    Config keys (``base_config``):

    - ``ai_engine``: mailbox AI engine (default ``GCP_AI_2``, AI Vision v3).
    - ``price_per_page_usd``: flat per-page price reported as ``cost_usd``.
    - ``poll_interval_s``: document polling cadence.
    - ``timeout_s``: budget per document from its upload, across runner retries.
    - ``max_requests_per_second``: client-side pacing shared by all threads (default 4,
      the API allows about 5).
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        api_key = os.getenv("PARSEUR_API_KEY")
        if not api_key:
            raise ProviderConfigError("PARSEUR_API_KEY environment variable is required for the parseur provider.")

        self._ai_engine = str(self.base_config.get("ai_engine", DEFAULT_AI_ENGINE))
        self._price_per_page = float(self.base_config.get("price_per_page_usd", 0.0))
        self._poll_interval = float(self.base_config.get("poll_interval_s", 5.0))
        # The slowest document of a full benchmark run took 1015 s. Past the runner's per-file
        # timeout (1800 s by default), its retry resumes polling the same document until this budget.
        self._timeout_s = float(self.base_config.get("timeout_s", 3600.0))
        rate = float(self.base_config.get("max_requests_per_second", 4.0))
        self._min_interval = 1.0 / rate if rate > 0 else 0.0
        self._rate_lock = threading.Lock()
        self._next_slot = 0.0

        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Token {api_key}"})
        self._session.mount("https://", HTTPAdapter(pool_maxsize=HTTP_POOL_SIZE))

        self._pool_lock = threading.Lock()
        self._pool: dict[str, int] = {}  # schema hash -> mailbox id
        self._pool_loaded = False

        # Per example_id: the stop flag of the running attempt, and the document it uploaded
        # (id, started_at, deadline) so a runner retry polls it again instead of uploading it twice.
        self._attempt_lock = threading.Lock()
        self._stops: dict[str, threading.Event] = {}
        self._uploads: dict[str, tuple[str, datetime, float]] = {}

    # ------------------------------------------------------------------ HTTP

    def _throttle(self) -> None:
        """Process-wide pacing so concurrent uploads and polls stay under the API rate limit."""
        if self._min_interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait > 0:
            time.sleep(wait)

    def _request(self, method: str, path: str, *, context: str, not_found_ok: bool = False, **kwargs: Any) -> Any:
        self._throttle()
        kwargs.setdefault("timeout", HTTP_TIMEOUT_S)
        try:
            response = self._session.request(method, f"{API_BASE_URL}{path}", **kwargs)
        except requests.RequestException as exc:
            raise ProviderTransientError(f"Network error during {context}: {exc}") from exc

        status = response.status_code
        if status == 404 and not_found_ok:
            return None
        if status == 429:
            raise ProviderRateLimitError(f"Rate limited during {context}: {response.text[:500]}")
        if status >= 500:
            raise ProviderTransientError(f"Server error {status} during {context}: {response.text[:500]}")
        if status >= 400:
            raise ProviderPermanentError(f"HTTP {status} during {context}: {response.text[:1000]}")
        if status == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderPermanentError(f"Non-JSON response during {context}: {response.text[:500]}") from exc

    # -------------------------------------------------------------- mailboxes

    def _load_pool(self) -> None:
        if self._pool_loaded:
            return
        page = 1
        while True:
            data = self._request("GET", "/parser", context="mailbox listing", params={"page": page})
            results = data.get("results") or []
            for mailbox in results:
                name = str(mailbox.get("name") or "")
                if name.startswith(MAILBOX_PREFIX) and mailbox.get("id") is not None:
                    self._pool.setdefault(name[len(MAILBOX_PREFIX) :], int(mailbox["id"]))
            current, total = data.get("current"), data.get("total")
            if not results or current is None or total is None or current >= total:
                break
            page += 1
        self._pool_loaded = True
        logger.info("Parseur mailbox pool: %d existing mailbox(es) with prefix %r", len(self._pool), MAILBOX_PREFIX)

    def _pooled_mailbox(self, schema_hash: str, conversion: dict[str, Any], instructions: str) -> int:
        with self._pool_lock:
            self._load_pool()
            if schema_hash in self._pool:
                return self._pool[schema_hash]
            name = f"{MAILBOX_PREFIX}{schema_hash}"
            payload: dict[str, Any] = {
                "name": name,
                "ai_engine": self._ai_engine,
                # Required for dotted field names (nested objects): without it the AI
                # engine drops them from the result. The result then comes back nested.
                "expand_result": True,
                "parser_object_set": conversion["parser_object_set"],
            }
            if instructions:
                payload["ai_instructions"] = instructions
            try:
                created = self._request("POST", "/parser", context="mailbox creation", json=payload)
            except ProviderTransientError:
                # The server may have created the mailbox before the error: list again on the retry.
                self._pool_loaded = False
                raise
            mailbox_id = int(created["id"])
            self._pool[schema_hash] = mailbox_id
            logger.info(
                "Created Parseur mailbox %s (%s) with %d top-level field(s)",
                mailbox_id,
                name,
                len(conversion["parser_object_set"]),
            )
            return mailbox_id

    # -------------------------------------------------------------- documents

    def _upload(self, mailbox_id: int, file_path: Path) -> str:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        files = {"file": (file_path.name, file_path.read_bytes(), mime_type)}
        try:
            data = self._request(
                "POST", f"/parser/{mailbox_id}/upload", context="file upload", files=files, timeout=UPLOAD_TIMEOUT_S
            )
        except ProviderTransientError as exc:
            # The server may have stored the file before the error: a retry would upload and bill it twice.
            # A rate-limited upload (HTTP 429) was rejected outright and stays retryable.
            raise ProviderPermanentError(f"Upload of {file_path.name} failed, not retried: {exc}") from exc
        # A file over the server's page limit is split into several documents: only the first is scored.
        attachments = (data or {}).get("attachments") or []
        if not attachments or not attachments[0].get("DocumentID"):
            raise ProviderPermanentError(f"Upload response did not contain a document id: {data!r}")
        return str(attachments[0]["DocumentID"])

    def _wait_for_document(
        self, document_id: str, deadline: float, stop: threading.Event
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Poll until the document is parsed; return it with its parsed result.

        The upload endpoint hands out the document id before the document is visible
        through the API, so a 404 counts as "not yet". A failure here is permanent: a
        retry would upload and bill the document again. ``stop`` is set by :meth:`cancel`.
        """
        document: dict[str, Any] | None = None
        poll_errors = 0
        empty_result_polls = 0
        # The first poll waits one interval: no document is done right after its upload.
        delay = self._poll_interval
        while not stop.wait(delay):
            delay = self._poll_interval
            try:
                fetched = self._request(
                    "GET", f"/document/{document_id}", context="document polling", not_found_ok=True
                )
                poll_errors = 0
            except ProviderRateLimitError:
                fetched = None
            except ProviderTransientError as exc:
                poll_errors += 1
                if poll_errors >= MAX_POLL_ERRORS:
                    raise ProviderPermanentError(
                        f"Document {document_id}: {poll_errors} polling errors in a row, last: {exc}",
                        job_id=document_id,
                        debug_payload=document,
                    ) from exc
                fetched = None
            if fetched is not None:
                document = fetched
                status = str(document.get("status") or "")
                if status in SUCCESS_STATUSES:
                    result = _parse_document_result(document)
                    if result is not None:
                        return document, result
                    empty_result_polls += 1
                    if empty_result_polls > MAX_EMPTY_RESULT_POLLS:
                        raise ProviderPermanentError(
                            f"Document {document_id} is {status} but has no result",
                            job_id=document_id,
                            debug_payload=document,
                        )
                    delay = min(EMPTY_RESULT_POLL_INTERVAL_S, self._poll_interval)
                elif status not in PENDING_STATUSES:
                    hint = " (account out of credits)" if status == "QUOTAEXC" else ""
                    raise ProviderPermanentError(
                        f"Document {document_id} ended with status {status}{hint}",
                        job_id=document_id,
                        debug_payload=document,
                    )
            if time.monotonic() >= deadline:
                last_status = document.get("status") if document else "not found"
                raise ProviderPermanentError(
                    f"Document {document_id} still '{last_status}' after {self._timeout_s:.0f}s",
                    job_id=document_id,
                    debug_payload=document,
                )
        raise ProviderTransientError(f"Polling of document {document_id} stopped by the runner", job_id=document_id)

    def _attach_cost_stats(self, raw_output: dict[str, Any]) -> None:
        num_pages = raw_output.get("num_pages")
        if isinstance(num_pages, int) and num_pages > 0 and self._price_per_page > 0:
            raw_output["cost_usd"] = num_pages * self._price_per_page
            raw_output["cost_per_page_usd"] = self._price_per_page
        credits = (raw_output.get("document") or {}).get("credits_used")
        if isinstance(credits, (int, float)):
            raw_output["credits_used"] = float(credits)

    # -------------------------------------------------------------- interface

    def cancel(self, example_id: str) -> bool:
        """Stop the poll loop of ``example_id`` after a runner timeout.

        The uploaded document is kept: the runner's retry polls it again instead of
        uploading and billing the file a second time.
        """
        with self._attempt_lock:
            stop = self._stops.get(example_id)
        if stop is None:
            return False
        stop.set()
        return True

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(f"Parseur provider only supports EXTRACT, got {request.product_type.value}")
        if not request.schema_override:
            raise ProviderPermanentError("Parseur provider requires schema_override (the document's JSON Schema)")

        file_path = Path(request.source_file_path)
        if not file_path.is_file():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        try:
            conversion = convert_json_schema(request.schema_override)
        except ValueError as exc:
            raise ProviderPermanentError(f"Cannot convert schema for {request.example_id}: {exc}") from exc
        if not conversion["parser_object_set"]:
            raise ProviderPermanentError(f"Schema for {request.example_id} maps to no Parseur field at all")
        if conversion["dropped"]:
            logger.warning(
                "%s: schema paths Parseur fields cannot represent, scored as null: %s",
                request.example_id,
                ", ".join(conversion["dropped"]),
            )

        # A JSON Schema's top-level description is its document-level instruction (which period, which
        # section, global conventions): the mailbox AI instructions in Parseur.
        description = request.schema_override.get("description")
        instructions = description.strip() if isinstance(description, str) else ""
        schema_hash = conversion_hash(conversion, ai_engine=self._ai_engine, ai_instructions=instructions)
        mailbox_id = self._pooled_mailbox(schema_hash, conversion, instructions)

        example_id = request.example_id
        stop = threading.Event()
        with self._attempt_lock:
            previous = self._stops.get(example_id)
            if previous is not None:
                previous.set()
            self._stops[example_id] = stop
            upload = self._uploads.get(example_id)
        try:
            if upload is None:
                # The clock starts at the upload: a mailbox is set up once per schema, as a customer does,
                # while the other providers take the schema with each request.
                started_at = datetime.now()
                deadline = time.monotonic() + self._timeout_s
                upload = (self._upload(mailbox_id, file_path), started_at, deadline)
                with self._attempt_lock:
                    self._uploads[example_id] = upload
            else:
                logger.info("%s: resuming polling of document %s", example_id, upload[0])
            document_id, started_at, deadline = upload
            try:
                document, result = self._wait_for_document(document_id, deadline, stop)
            except ProviderError:
                if not stop.is_set():
                    with self._attempt_lock:
                        self._uploads.pop(example_id, None)
                raise
            with self._attempt_lock:
                self._uploads.pop(example_id, None)
        finally:
            with self._attempt_lock:
                if self._stops.get(example_id) is stop:
                    self._stops.pop(example_id)

        completed_at = datetime.now()
        document["result"] = result
        # The document text and OCR pages are large and not needed to re-normalize.
        for heavy_key in ("content", "ocr_page_set"):
            document.pop(heavy_key, None)

        raw_output: dict[str, Any] = {
            "document": document,
            "document_id": document_id,
            "mailbox_id": mailbox_id,
            "schema_hash": schema_hash,
            "ai_engine": self._ai_engine,
            "conversion": conversion,
            "num_pages": page_count(file_path),
        }

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        raw_output = dict(raw_result.raw_output)
        result = _parse_document_result(raw_output.get("document") or {})
        conversion = raw_output.get("conversion")
        if result is None:
            raise ProviderPermanentError("Parseur document has no result to normalize")
        if not isinstance(conversion, dict) or "fields" not in conversion:
            raise ProviderPermanentError("raw_output is missing the schema conversion needed to decode the result")

        self._attach_cost_stats(raw_output)
        output = ExtractOutput(
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=decode_result(result, conversion),
            field_citations=[],
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
