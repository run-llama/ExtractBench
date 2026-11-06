"""Registration stays lazy across the extension, helper and provider-factory seams."""

import subprocess
import sys
import textwrap


def test_helpers_do_not_load_providers_and_factory_loads_builtins_once() -> None:
    """Catch eager SDK imports, missing factory registration and repeated missing-SDK imports."""
    probe = textwrap.dedent(
        """
        import importlib
        import sys
        from unittest.mock import patch

        import extract_bench.extensions
        import extract_bench.evaluation.layout_adapters
        import extract_bench.evaluation.layout_label_mappers
        from extract_bench.inference.providers import extract, layoutdet, parse
        from extract_bench.inference.providers.registry import create_provider
        from extract_bench.schemas.pipeline import PipelineSpec

        implementation_modules = {
            f"{package.__name__}.{module}"
            for package in (extract, layoutdet, parse)
            for module in package._PROVIDER_MODULES
        }
        loaded = sorted(implementation_modules.intersection(sys.modules))
        assert loaded == [], loaded

        pipeline = PipelineSpec(
            pipeline_name="lazy_registration_probe",
            provider_name="qwen3_5",
            product_type="parse",
            config={"server_url": "https://example.invalid"},
        )
        with patch(
            "extract_bench.inference.providers.registry.importlib.import_module",
            wraps=importlib.import_module,
        ) as imports:
            provider = create_provider(pipeline)
            assert type(provider).__name__ == "Qwen35Provider"
            first_load_calls = imports.call_count
            assert first_load_calls > 0
            assert type(create_provider(pipeline)) is type(provider)
            assert imports.call_count == first_load_calls
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
