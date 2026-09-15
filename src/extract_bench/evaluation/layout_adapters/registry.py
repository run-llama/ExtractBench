"""Decorator-driven registry for layout adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from extract_bench.evaluation.layout_adapters.base import LayoutAdapter
from extract_bench.inference.pipelines import get_pipeline
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceResult

_DEFAULT_LAYOUT_ADAPTER_KEY = "__default__"


@dataclass(frozen=True)
class _LayoutAdapterRegistration:
    keys: tuple[str, ...]
    priority: int
    adapter_cls: type[LayoutAdapter]


_LAYOUT_ADAPTER_REGISTRY: list[_LayoutAdapterRegistration] = []
_adapters_loaded = False


def _ensure_adapters_loaded() -> None:
    """
    Import concrete adapter modules once so decorators populate the registry,
    which can be done lazily after other imports as long as it takes place
    before the first registry read.
    """
    global _adapters_loaded
    if _adapters_loaded:
        return
    # Set the flag before importing so registrations made while adapters is
    # importing (which call back into this module) no-op instead of re-entering
    # a partially initialized import.
    _adapters_loaded = True
    snapshot = list(_LAYOUT_ADAPTER_REGISTRY)
    try:
        from . import adapters as _adapters  # noqa: F401
    except BaseException:
        # Roll back partial registrations so a retried import starts clean and
        # surfaces the original error instead of duplicate-key failures.
        _adapters_loaded = False
        _LAYOUT_ADAPTER_REGISTRY[:] = snapshot
        raise


def _registrations() -> Sequence[_LayoutAdapterRegistration]:
    """Return registered adapters, loading bindings on first access."""
    _ensure_adapters_loaded()
    return _LAYOUT_ADAPTER_REGISTRY


def _get_highest_priority_registration(
    registrations: Sequence[_LayoutAdapterRegistration],
) -> _LayoutAdapterRegistration:
    """Return the highest-priority registration from a non-empty sequence."""
    return max(registrations, key=lambda entry: entry.priority)


def register_layout_adapter(
    *provider_keys: str,
    priority: int = 0,
) -> Callable[[type[LayoutAdapter]], type[LayoutAdapter]]:
    """Register a layout adapter class for one or more provider keys."""
    if not provider_keys:
        raise ValueError("register_layout_adapter requires at least one provider key")

    def decorator(cls: type[LayoutAdapter]) -> type[LayoutAdapter]:
        # Load the built-in adapters first so the duplicate-key check below sees
        # them even when an external registration happens before the first
        # registry read. No-ops while adapters.py itself is importing.
        _ensure_adapters_loaded()
        existing_keys = {key for entry in _LAYOUT_ADAPTER_REGISTRY for key in entry.keys}
        for key in provider_keys:
            if key in existing_keys:
                raise ValueError(f"Layout adapter already registered for provider key '{key}'")

        _LAYOUT_ADAPTER_REGISTRY.append(
            _LayoutAdapterRegistration(
                keys=tuple(provider_keys),
                priority=priority,
                adapter_cls=cls,
            )
        )
        return cls

    return decorator


def list_layout_adapters() -> list[str]:
    """List all registered adapter keys."""
    keys = {key for registration in _registrations() for key in registration.keys}
    return sorted(keys)


PipelineResolver = Callable[[str], PipelineSpec | None]

_PIPELINE_RESOLVERS: list[PipelineResolver] = []


def register_pipeline_resolver(resolver: PipelineResolver) -> None:
    """Let a harness resolve pipeline names the package registry does not know.

    ``resolve_layout_provider_name`` maps a result's ``pipeline_name`` to the
    provider key the adapter and label-mapper registries are keyed on. A
    downstream harness that keeps its own pipeline registry registers a
    ``name -> PipelineSpec | None`` callable here; resolvers are consulted
    before the package registry, so the harness's definition of a shared
    name wins.
    """
    if resolver not in _PIPELINE_RESOLVERS:
        _PIPELINE_RESOLVERS.append(resolver)


def resolve_layout_provider_name(inference_result: InferenceResult) -> str | None:
    """Resolve provider key from pipeline metadata when available."""
    pipeline_name = inference_result.pipeline_name
    for resolver in _PIPELINE_RESOLVERS:
        try:
            spec = resolver(pipeline_name)
        except Exception:
            continue
        if spec is not None:
            return spec.provider_name
    try:
        pipeline_spec = get_pipeline(pipeline_name)
        return pipeline_spec.provider_name
    except Exception:
        return None


def create_layout_adapter_for_provider(provider_name: str) -> LayoutAdapter:
    """
    Instantiate a layout adapter for a provider key, or raise a ValueError if no
    matching registration is found.
    """
    candidates = tuple(registration for registration in _registrations() if provider_name in registration.keys)
    if len(candidates) > 0:
        return _get_highest_priority_registration(candidates).adapter_cls()

    available = ", ".join(list_layout_adapters())
    raise ValueError(f"No layout adapter registered for provider '{provider_name}'. Available adapters: {available}")


def create_layout_adapter_for_result(inference_result: InferenceResult) -> LayoutAdapter:
    """Resolve and instantiate adapter using provider key first, matcher fallback second."""
    provider_name = resolve_layout_provider_name(inference_result)
    if provider_name is not None:
        try:
            return create_layout_adapter_for_provider(provider_name)
        except ValueError:
            # A known provider without a registered adapter lands on the default
            # adapter. Shape matchers cannot distinguish providers here: many
            # emit the same shared layout IR, so a matcher would silently score
            # one provider's results with another provider's adapter. Matchers
            # only run below, for results whose pipeline cannot be resolved.
            return create_layout_adapter_for_provider(_DEFAULT_LAYOUT_ADAPTER_KEY)

    candidates: list[_LayoutAdapterRegistration] = []
    for registration in _registrations():
        # only consider provider-specific registrations
        if _DEFAULT_LAYOUT_ADAPTER_KEY in registration.keys:
            continue
        if registration.adapter_cls.matches(inference_result):
            candidates.append(registration)
    if len(candidates) > 0:
        return _get_highest_priority_registration(candidates).adapter_cls()

    return create_layout_adapter_for_provider(_DEFAULT_LAYOUT_ADAPTER_KEY)
