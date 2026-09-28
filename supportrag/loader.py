"""製品資料（Markdown）を、見出し単位の検索用チャンクに分ける。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_title: str
    heading: str
    text: str
    path: str

    @property
    def title(self) -> str:
        return f"{self.doc_title}｜{self.heading}"


def _split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """H1を資料名、H2ごとを1チャンクにする。H2より前の本文は「概要」として扱う。"""
    doc_title = ""
    sections: list[tuple[str, list[str]]] = [("概要", [])]
    for line in markdown.splitlines():
        if line.startswith("# ") and not doc_title:
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)
    return doc_title, [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        doc_title, sections = _split_sections(path.read_text(encoding="utf-8"))
        doc_title = doc_title or path.stem
        for index, (heading, text) in enumerate(sections):
            chunks.append(Chunk(f"{path.stem}#{index}", doc_title, heading, text, path.name))
    return chunks


def fingerprint(chunks: list[Chunk]) -> str:
    """資料が変わったときだけ埋め込みを作り直すための値。"""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.id}\0{chunk.title}\0{chunk.text}\0".encode())
    return digest.hexdigest()
