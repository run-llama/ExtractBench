from __future__ import annotations

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required")

from extract_bench.inference.providers.extract import table_codegen_extract as tce


def _provider(model: str) -> tce.TableCodegenExtractProvider:
    provider = object.__new__(tce.TableCodegenExtractProvider)
    provider._model = model
    provider._pricing = dict(tce._PRICING_PER_1M)
    provider._llm_provider = "anthropic"
    return provider


def test_claude_codegen_cost_uses_listed_cache_rates() -> None:
    # Sonnet 5 listed rates: $2 in, $10 out, $0.20 cache read, $2.50 5m cache write.
    usage = {"input": 1_000_000, "output": 1_000_000, "cache_read": 1_000_000, "cache_write": 1_000_000}
    assert _provider("claude-sonnet-5")._codegen_cost(usage) == pytest.approx(2.0 + 10.0 + 0.20 + 2.50)
