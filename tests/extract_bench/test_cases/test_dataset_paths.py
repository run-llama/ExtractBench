from __future__ import annotations

from extract_bench.test_cases.dataset_paths import (
    dataset_name_from_metadata,
    dataset_name_from_test_cases_dir,
)


def test_dataset_name_from_test_cases_dir_nested_path() -> None:
    path = "/srv/bench/data/extract/short/v0.2"
    assert dataset_name_from_test_cases_dir(path) == "extract/short/v0.2"


def test_dataset_name_from_test_cases_dir_single_segment() -> None:
    assert dataset_name_from_test_cases_dir("/tmp/financial_tables") == "financial_tables"


def test_dataset_name_from_metadata() -> None:
    metadata = {"test_cases_dir": "/srv/bench/data/tables_core/v1.0"}
    assert dataset_name_from_metadata(metadata) == "tables_core/v1.0"


def test_dataset_name_from_metadata_missing_dir() -> None:
    assert dataset_name_from_metadata({}) == "unknown"


def test_parse_dataset_name_plain_base_and_version() -> None:
    from extract_bench.test_cases.dataset_paths import parse_dataset_name

    assert parse_dataset_name("extract/short/v0.2") == ("extract/short", "v0.2")
    assert parse_dataset_name("tables_core") == ("tables_core", None)
    assert parse_dataset_name("unknown") == ("unknown", None)


def test_parse_dataset_name_collection_folder_keeps_the_dataset_as_base() -> None:
    """``parse_features/`` is a folder of datasets, each with its own versions - not one dataset."""
    from extract_bench.test_cases.dataset_paths import parse_dataset_name

    assert parse_dataset_name("parse_features/inline_images/v0.2") == ("parse_features/inline_images", "v0.2")
    assert parse_dataset_name("parse_features/inline_images/v0.1") == ("parse_features/inline_images", "v0.1")
    assert parse_dataset_name("extract/tables/v0.2") == ("extract/tables", "v0.2")
    # a sub-folder inside the version directory stays part of the version
    assert parse_dataset_name("extract/redline_qa/v0.3/qa") == ("extract/redline_qa", "v0.3/qa")


def test_parse_dataset_name_without_version_segment_keeps_legacy_split() -> None:
    from extract_bench.test_cases.dataset_paths import parse_dataset_name

    assert parse_dataset_name("split_tests/real-docs-v1") == ("split_tests", "real-docs-v1")
    assert parse_dataset_name("mckesson/anonymized") == ("mckesson", "anonymized")
