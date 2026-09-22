"""ExtractBench adapter for LiteParse + Jev source-selection pipelines."""

import importlib
import os
from datetime import datetime
from pathlib import Path

from extract_bench.inference.providers.base import Provider, ProviderConfigError, ProviderPermanentError
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType

VARIANTS = (
    "fields",
    "hierarchical",
    "tables",
    "geometric",
    "localized",
    "multipage",
    "hybrid",
    "boundaries",
    "rowrepair",
    "compact",
    "consensus",
    "anchors",
    "router",
)


@register_provider("jev_liteparse")
class JevLiteParseProvider(Provider):
    """Local source candidates, Jev typed decisions, deterministic JSON assembly."""

    def __init__(self, provider_name, base_config=None):
        super().__init__(provider_name, base_config)
        self.variant = self.base_config.get("variant", "router")
        if self.variant not in VARIANTS:
            raise ProviderConfigError(f"Unknown Jev variant: {self.variant}")

    def run_inference(self, pipeline, request):
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError("Jev supports extraction only")
        if request.schema_override is None:
            raise ProviderConfigError("Jev requires a public JSON schema")
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ProviderConfigError("Set OPENROUTER_API_KEY")
        from .jev.citations import build_citations
        from .jev.client import JevClient
        from .jev.parser import parse_document

        root = Path(self.base_config.get("artifact_directory", "output/jev_liteparse"))
        started = datetime.now()
        client = JevClient(
            root / "api",
            label=f"{pipeline.pipeline_name}/{request.example_id}",
            budget=float(self.base_config.get("budget_usd", 9)),
            model=self.base_config.get("model", "typesafe/jev-1.13"),
        )
        try:
            document = parse_document(request.source_file_path, root / "parse_cache")
            module = importlib.import_module("extract_bench.inference.providers.extract.jev.variant_" + self.variant)
            prediction = module.extract(document, request.schema_override, client)
            if not isinstance(prediction.get("data"), (dict, list)) or not prediction["data"]:
                raise ProviderPermanentError("Jev pipeline returned no structured extraction")
            citations = build_citations(prediction, document)
        except ProviderPermanentError:
            raise
        except Exception as exc:
            # No blanket retries: the durable ledger retains uncertain request costs.
            raise ProviderPermanentError(f"Jev extraction failed: {exc}") from exc
        completed = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=ProductType.EXTRACT,
            raw_output={
                "prediction": prediction,
                "field_citations": citations,
                "cost_usd": client.cost,
                "cost_per_page_usd": client.cost / max(1, len(document["pages"])),
                "num_pages": len(document["pages"]),
                "num_api_calls": client.calls,
                "model": client.model,
                "parser": "liteparse",
                "parser_version": document["parser_version"],
            },
            started_at=started,
            completed_at=completed,
            latency_in_ms=int((completed - started).total_seconds() * 1000),
        )

    def normalize(self, raw_result):
        raw = raw_result.raw_output
        output = ExtractOutput(
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=raw["prediction"]["data"],
            field_citations=[FieldCitation(**c) for c in raw.get("field_citations", [])],
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=ProductType.EXTRACT,
            raw_output=raw,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
