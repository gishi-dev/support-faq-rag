"""日本語向けのハイブリッド検索（キーワード検索＋意味検索）。

キーワード検索は文字の2文字区切り（バイグラム）でBM25を計算する。
日本語は単語の間に空白がないため、空白区切りを前提にした全文検索では語が拾えない。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .loader import Chunk, fingerprint

_CJK = r"぀-ヿ㐀-鿿豈-﫿"
_TOKEN = re.compile(rf"[{_CJK}ー]+|[a-z0-9]+")
_HIRAGANA_ONLY = re.compile(r"[぀-ゟー]+")


def tokenize(text: str) -> list[str]:
    """ひらがなだけの2文字（「にな」「てい」など）は、どの資料にも出るため除く。"""
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for run in _TOKEN.findall(text):
        if run.isascii() or len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return [t for t in tokens if not _HIRAGANA_ONLY.fullmatch(t)]


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.freqs = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avg_length = sum(self.lengths) / max(len(documents), 1)
        df = Counter(term for doc in documents for term in set(doc))
        n = len(documents)
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        result = np.zeros(len(self.freqs))
        for i, (freq, length) in enumerate(zip(self.freqs, self.lengths)):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            result[i] = sum(
                self.idf[t] * freq[t] * (self.k1 + 1) / (freq[t] + norm) for t in set(query) if t in freq
            )
        return result


@dataclass
class Hit:
    chunk: Chunk
    score: float
    keyword_rank: int | None
    vector_rank: int | None


def _load_embedder(model: str, cache_dir: Path):
    """モデルをプロジェクト内へ実ファイルで保存してから読み込む。

    fastembed既定のキャッシュはシンボリックリンクで、外部データを持つ大きなONNXモデルは
    onnxruntimeのパス検査で読み込めないことがある。
    """
    from fastembed import TextEmbedding
    from huggingface_hub import snapshot_download

    info = next(m for m in TextEmbedding.list_supported_models() if m["model"] == model)
    path = cache_dir / "models" / model.split("/")[-1]
    if not (path / info["model_file"]).exists():
        snapshot_download(info["sources"]["hf"], local_dir=path)
    return TextEmbedding(model, specific_model_path=str(path))


class HybridSearcher:
    def __init__(self, chunks: list[Chunk], embed_model: str, cache_dir: Path) -> None:
        self.chunks = chunks
        self.bm25 = BM25([tokenize(f"{c.title}\n{c.text}") for c in chunks])
        self.embedder = _load_embedder(embed_model, cache_dir)
        # e5系は、質問と資料で決まった前置きを付けて学習されている。
        self.prefixes = ("query: ", "passage: ") if "e5" in embed_model else ("", "")
        self.vectors = self._load_or_embed(embed_model, cache_dir)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self.embedder.embed(texts)), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def _load_or_embed(self, embed_model: str, cache_dir: Path) -> np.ndarray:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = f"{embed_model}\0{fingerprint(self.chunks)}"
        cache = cache_dir / "embeddings.npz"
        if cache.exists():
            stored = np.load(cache, allow_pickle=False)
            if str(stored["key"]) == key:
                return stored["vectors"]
        vectors = self._embed([f"{self.prefixes[1]}{c.title}\n{c.text}" for c in self.chunks])
        np.savez(cache, key=np.array(key), vectors=vectors)
        return vectors

    def search(
        self, query: str, top_k: int = 5, vector_weight: float = 0.6, keyword_weight: float = 0.4, rrf_k: int = 10
    ) -> list[Hit]:
        """両方の順位を重み付きの Reciprocal Rank Fusion で合わせる。点数の尺度が違っても混ぜられる。

        重みと rrf_k は、20問の確認用質問で上位5件に正解が入る数が最も多かった値。
        """
        keyword = self.bm25.scores(tokenize(query))
        vector = self.vectors @ self._embed([self.prefixes[0] + query])[0]
        keyword_order = [i for i in np.argsort(-keyword) if keyword[i] > 0]
        vector_order = list(np.argsort(-vector))
        keyword_rank = {i: r for r, i in enumerate(keyword_order, 1)}
        vector_rank = {i: r for r, i in enumerate(vector_order, 1)}
        fused = {
            i: vector_weight / (rrf_k + vector_rank[i])
            + (keyword_weight / (rrf_k + keyword_rank[i]) if i in keyword_rank else 0)
            for i in range(len(self.chunks))
        }
        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [Hit(self.chunks[i], fused[i], keyword_rank.get(i), vector_rank[i]) for i in best]
