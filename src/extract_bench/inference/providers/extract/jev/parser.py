"""Local LiteParse source text and geometry, cached by PDF and parser version."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def parse_document(pdf, cache):
    import liteparse

    pdf, cache = Path(pdf), Path(cache)
    version = importlib.metadata.version("liteparse")
    digest = hashlib.sha256(pdf.read_bytes() + version.encode() + b"jev-markdown-words-v2").hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (digest + ".json")
    if target.exists():
        return json.loads(target.read_text())
    result = liteparse.LiteParse(output_format="markdown", quiet=True, emit_word_boxes=True, extract_links=False).parse(
        str(pdf)
    )
    pages = []
    for page in result.pages:
        words = []
        for item in page.text_items:
            for word in item.words or [item]:
                words.append({"text": word.text, "x": word.x, "y": word.y, "width": word.width, "height": word.height})
        pages.append(
            {
                "page_index": page.page_num - 1,
                "text": page.text,
                "markdown": page.markdown,
                "words": words,
                "width": page.width,
                "height": page.height,
            }
        )
    document = {
        "pages": pages,
        "text": "\n\n".join(p["text"] for p in pages),
        "markdown": "\n\n".join(p["markdown"] for p in pages),
        "parser": "liteparse",
        "parser_version": version,
        "sha256": digest,
    }
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False))
    temporary.replace(target)
    return document
