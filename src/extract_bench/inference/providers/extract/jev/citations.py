"""Citations constrained to extractor evidence, never a guessed global occurrence."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation


def _path(data, path):
    """Resolve and normalize bracket/dot array indices against actual data types."""
    value, canonical = data, ""
    try:
        for part in re.findall(r"[^.\[\]]+", str(path).strip(".")):
            if isinstance(value, list):
                index = int(part)
                if index < 0:
                    return None, None
                value = value[index]
                canonical += f"[{index}]"
            else:
                value = value[part]
                canonical += ("." if canonical else "") + part
        return canonical, value
    except (KeyError, IndexError, ValueError, TypeError):
        return None, None


def _compact(value):
    return re.sub(r"[^\w]", "", str(value).casefold())


def _number(value):
    value = re.sub(r"[$€£¥,%\s]", "", str(value))
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    try:
        number = Decimal(value)
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def _supported(value, source):
    if isinstance(value, (int, float)):
        expected = _number(value)
        return any(
            _number(match.group()) == expected
            for match in re.finditer(r"\(?[-+]?[$€£¥]?\d[\d,]*(?:\.\d+)?%?\)?", str(source))
        )
    return bool(_spans([{"text": word} for word in str(source).split()], value))


def _ordered_words(page):
    words = [
        w
        for w in page.get("words", [])
        if w.get("text")
        and all(isinstance(w.get(k), (int, float)) and math.isfinite(w[k]) for k in ("x", "y", "width", "height"))
    ]
    rows = []
    for word in sorted(words, key=lambda w: (w["y"], w["x"])):
        if not rows or abs(word["y"] - rows[-1][0]["y"]) > max(1, min(word["height"], rows[-1][0]["height"]) * 0.5):
            rows.append([word])
        else:
            rows[-1].append(word)
    return [word for row in rows for word in sorted(row, key=lambda w: w["x"])]


def _spans(words, target, numeric=False):
    """Exact whole-word spans, so '12' cannot cite a word containing '120'."""
    expected = _number(target) if numeric else _compact(target)
    if expected is None or expected == "":
        return []
    matches = []
    # Numeric source representations can add punctuation and trailing zeros.
    budget = max(40, len(str(target)) + 20) if numeric else len(expected)
    for start in range(len(words)):
        raw = ""
        for end in range(start, len(words)):
            raw += (" " if raw else "") + words[end]["text"]
            normalized = _number(raw) if numeric else _compact(raw)
            if normalized == expected:
                matches.append((start, end + 1))
                break
            if len(_compact(raw)) > budget:
                break
    return matches


def build_citations(prediction, document):
    citations, seen = [], set()
    # Call-local caches cannot outlive or become stale against a document. Cache
    # value types and spellings separately to preserve numeric matching budgets.
    word_cache, page_text_cache, page_match_cache = {}, {}, {}
    support_cache, span_cache, local_span_cache = {}, {}, {}

    def words_for(page):
        identity = id(page)
        if identity not in word_cache:
            word_cache[identity] = _ordered_words(page)
        return word_cache[identity]

    def text_for(page):
        identity = id(page)
        if identity not in page_text_cache:
            raw = page.get("text", "") or " ".join(w.get("text", "") for w in words_for(page))
            page_text_cache[identity] = _compact(raw)
        return page_text_cache[identity]

    for item in prediction.get("evidence", []):
        raw_path = item.get("field_path", item.get("path", item.get("field")))
        if not raw_path:
            continue
        path, value = _path(prediction.get("data"), raw_path)
        if not path or value is None or isinstance(value, (bool, dict, list)):
            continue
        source = item.get("source_text", item.get("text", ""))
        context = item.get("context") or source
        context = str(context)
        page_index = item.get("page_index")
        # Localized variant prefixes human-readable page metadata to source context.
        metadata = re.match(r"^Page (\d+), lines around \d+:\n", context)
        if metadata:
            if page_index is None:
                page_index = int(metadata[1]) - 1
            context = context[metadata.end() :]
        value_key = (type(value), str(value))
        support_key = (value_key, context)
        if context and support_key not in support_cache:
            support_cache[support_key] = _supported(value, context)
        if not context or not support_cache[support_key]:
            continue
        page_match_key = (page_index, context)
        if page_match_key not in page_match_cache:
            normalized = _compact(context)
            page_match_cache[page_match_key] = [
                page
                for page in document.get("pages", [])
                if (page_index is None or page.get("page_index") == page_index)
                and normalized
                and normalized in text_for(page)
            ]
        pages = page_match_cache[page_match_key]
        if len(pages) != 1:
            continue
        page = pages[0]
        citation = {
            "field_path": path,
            "page": page["page_index"] + 1,
            "reference_text": str(value),
            "source": "jev_liteparse",
        }
        words = words_for(page)
        context_key = (id(page), context, False)
        if context_key not in span_cache:
            span_cache[context_key] = _spans(words, context)
        context_spans = span_cache[context_key]
        matches = []
        for start, end in context_spans:
            local_key = (id(page), start, end, value_key, isinstance(value, (int, float)))
            if local_key not in local_span_cache:
                local_span_cache[local_key] = _spans(words[start:end], value, isinstance(value, (int, float)))
            for local_start, local_end in local_span_cache[local_key]:
                span = (start + local_start, start + local_end)
                # Explicit physical row coordinates disambiguate repeated local values.
                if "y" in item and isinstance(item["y"], (int, float)):
                    if not all(abs(w["y"] - item["y"]) <= max(1, w["height"] * 0.6) for w in words[span[0] : span[1]]):
                        continue
                matches.append(span)
        matches = sorted(set(matches))
        width, height = page.get("width", 0), page.get("height", 0)
        emitted = []
        if len(matches) == 1 and width > 0 and height > 0:
            for word in words[matches[0][0] : matches[0][1]]:
                x0, y0 = max(0, word["x"]), max(0, word["y"])
                x1, y1 = min(width, word["x"] + word["width"]), min(height, word["y"] + word["height"])
                if x1 > x0 and y1 > y0:
                    emitted.append(
                        {**citation, "bbox": [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]}
                    )
        for result in emitted or [citation]:
            key = (path, result["page"], tuple(result.get("bbox", [])), result["reference_text"])
            if key not in seen:
                citations.append(result)
                seen.add(key)
    return citations
