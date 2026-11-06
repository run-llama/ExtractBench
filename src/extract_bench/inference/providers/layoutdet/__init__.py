"""Layout detection providers loaded by the registry on first use."""

import importlib
import logging

logger = logging.getLogger(__name__)

_PROVIDER_MODULES = [
    "chandra",
    "docling",
    "dots_ocr",
    "layout_v3",
    "layout_v3_byoc",
    "paddle",
    "qwen3vl",
    "surya",
    "yolo",
]


def load_providers() -> None:
    """Register this family's providers, skipping optional SDKs that are unavailable."""
    for module in _PROVIDER_MODULES:
        try:
            importlib.import_module(f"extract_bench.inference.providers.layoutdet.{module}")
        except ImportError as exc:
            # A missing third-party SDK is expected on a core-only install; a
            # failure originating inside this package is a packaging bug and must
            # be loud, not silently skipped.
            if exc.name is not None and exc.name.startswith("extract_bench"):
                logger.warning("Layout provider %s failed to import: %s", module, exc)
            else:
                logger.debug("Skipping layout provider %s (missing dependency)", module)


__all__ = _PROVIDER_MODULES
