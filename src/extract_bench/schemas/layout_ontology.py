"""Canonical layout ontology and label sets.

Re-exported from ``parse-bench``: the ontology is shared vocabulary, and the
label enums are compared by identity across package boundaries.
"""

from parse_bench.schemas.layout_ontology import (
    BASIC_LABELS,
    CANONICAL_TO_BASIC,
    CANONICAL_TO_CORE,
    CORE_LABELS,
    DEFAULT_LAYOUT_EVALUATION_ONTOLOGY,
    DOCLAYNET_ID_TO_LABEL,
    SUPPORTED_LAYOUT_EVALUATION_ONTOLOGIES,
    BasicLabel,
    CanonicalLabel,
    CanonicalLayoutDetectionOntology,
    CoreLayoutDetectionOntology,
    LayoutDetectionOntology,
    OntologyType,
    get_ontology,
)

__all__ = [
    "BASIC_LABELS",
    "CANONICAL_TO_BASIC",
    "CANONICAL_TO_CORE",
    "CORE_LABELS",
    "DEFAULT_LAYOUT_EVALUATION_ONTOLOGY",
    "DOCLAYNET_ID_TO_LABEL",
    "SUPPORTED_LAYOUT_EVALUATION_ONTOLOGIES",
    "BasicLabel",
    "CanonicalLabel",
    "CanonicalLayoutDetectionOntology",
    "CoreLayoutDetectionOntology",
    "LayoutDetectionOntology",
    "OntologyType",
    "get_ontology",
]
