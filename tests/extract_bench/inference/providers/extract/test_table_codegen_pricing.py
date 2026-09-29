from __future__ import annotations

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required")

from extract_bench.inference.providers.extract import table_codegen_extract as tce


def _provider(model: str, **base_config: object) -> tce.TableCodegenExtractProvider:
    return tce.TableCodegenExtractProvider("table_codegen_extract", {"model": model, **base_config})


def test_claude_codegen_cost_uses_listed_cache_rates() -> None:
    # Sonnet 5 listed rates: $2 in, $10 out, $0.20 cache read, $2.50 5m cache write.
    usage = {"input": 1_000_000, "output": 1_000_000, "cache_read": 1_000_000, "cache_write": 1_000_000}
    assert _provider("claude-sonnet-5")._codegen_cost(usage) == pytest.approx(2.0 + 10.0 + 0.20 + 2.50)


def test_custom_input_output_pricing_keeps_listed_cache_rates() -> None:
    # Opus 5.5 is not in the default in/out table; custom pricing must not zero its cache cost.
    provider = _provider("claude-opus-5-5", pricing={"claude-opus-5-5": (4.0, 20.0)})
    usage = {"cache_read": 1_000_000, "cache_write": 1_000_000}
    assert provider._codegen_cost(usage) == pytest.approx(0.20 + 5.0)


def test_cache_pricing_override() -> None:
    provider = _provider("claude-sonnet-5", cache_pricing={"claude-sonnet-5": (0.5, 3.0)})
    usage = {"cache_read": 1_000_000, "cache_write": 1_000_000}
    assert provider._codegen_cost(usage) == pytest.approx(0.5 + 3.0)
