"""Public extension points for building a benchmark harness on top of extract-bench.

Everything a downstream package needs to add its own providers, pipelines,
layout adapters and evaluators without forking extract-bench:

- :func:`register_provider` / :func:`register_pipeline` — new extraction
  systems and their configurations.
- :func:`register_layout_adapter` / :func:`register_layout_label_mapper` — how a
  provider's layout output maps onto the canonical ontology used for word-level
  grounding.
- ``EvaluationRunner.register_evaluator(product_type, evaluator)`` — scoring for
  a product type.

Registrations take effect when the extension module is imported, so an
extension package typically performs them in its top-level ``__init__``::

    from extract_bench.extensions import register_pipeline, register_provider
    from extract_bench.inference.providers.base import Provider
    from extract_bench.schemas.pipeline import PipelineSpec
    from extract_bench.schemas.product import ProductType

    @register_provider("my_extractor")
    class MyExtractor(Provider):
        ...

    register_pipeline(
        PipelineSpec(
            pipeline_name="my_extractor_default",
            provider_name="my_extractor",
            product_type=ProductType.EXTRACT,
            config={"model": "v1"},
        )
    )

The CLI is a Google Fire class; subclass :class:`extract_bench.cli.BenchCLI` and
add attributes for extra command groups.
"""

from extract_bench.evaluation.layout_adapters.registry import register_layout_adapter
from extract_bench.evaluation.layout_label_mappers.registry import register_layout_label_mapper
from extract_bench.inference.pipelines import register_pipeline
from extract_bench.inference.providers.registry import register_provider

__all__ = [
    "register_layout_adapter",
    "register_layout_label_mapper",
    "register_pipeline",
    "register_provider",
]
