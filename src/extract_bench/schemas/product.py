"""Product types: the kind of task a pipeline performs.

Re-exported from ``parse-bench``. Sharing the module also shares the extension
registry behind :func:`register_product_type`, so a product a downstream harness
registers is accepted by both packages' ``ProductType`` fields.
"""

from parse_bench.schemas.product import (
    ExtensionProductType,
    ProductType,
    ProductTypeName,
    coerce_product_type,
    register_product_type,
    registered_product_types,
)

__all__ = [
    "ExtensionProductType",
    "ProductType",
    "ProductTypeName",
    "coerce_product_type",
    "register_product_type",
    "registered_product_types",
]
