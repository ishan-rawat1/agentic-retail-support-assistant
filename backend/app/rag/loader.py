"""Knowledge-base document loading and normalization."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class KBDocument:
    doc_id: str          # stable id derived from filename, e.g. "returns-policy"
    source: str          # original filename
    title: str
    category: str
    text: str
    metadata: dict = field(default_factory=dict)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Parse a minimal ``key: value`` YAML frontmatter block."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, raw[match.end():]


def normalize_text(text: str) -> str:
    """Normalize whitespace without destroying Markdown structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)      # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank runs
    return text.strip()


def load_documents(docs_dir: str | Path) -> list[KBDocument]:
    """Load all Markdown knowledge-base documents from a directory."""
    docs_dir = Path(docs_dir)
    documents: list[KBDocument] = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        doc = KBDocument(
            doc_id=path.stem,
            source=path.name,
            title=meta.get("title", path.stem.replace("-", " ").title()),
            category=meta.get("category", "general"),
            text=normalize_text(body),
            metadata=meta,
        )
        documents.append(doc)
    logger.info("Loaded %d knowledge-base documents from %s", len(documents), docs_dir)
    return documents
