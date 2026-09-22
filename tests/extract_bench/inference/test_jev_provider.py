import pytest

from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.base import ProviderConfigError, ProviderPermanentError
from extract_bench.inference.providers.registry import create_provider
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType


def test_provider_registration_and_input_contract():
    pipeline = get_pipeline("jev_liteparse_hybrid")
    provider = create_provider(pipeline)
    request = InferenceRequest(example_id="x", source_file_path="x.pdf", product_type=ProductType.PARSE)
    with pytest.raises(ProviderPermanentError, match="extraction only"):
        provider.run_inference(pipeline, request)
    request.product_type = ProductType.EXTRACT
    with pytest.raises(ProviderConfigError, match="JSON schema"):
        provider.run_inference(pipeline, request)
