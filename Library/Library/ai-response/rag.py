from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Sequence

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from knowledge_base import (
    LEGAL_DOCS_DIR,
    ensure_knowledge_base,
    load_knowledge_base_documents,
)


ARTICLE_PATTERN = re.compile(r"(\d+)-modda")
DEFAULT_CHUNK_WORDS = 650
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 4
MAX_FEATURES = 4096

SOURCE_TITLE_OVERRIDES = {
    "constitution.txt": "Konstitutsiya",
    "bank_law.txt": "Bank qonuni",
    "tax_code.txt": "Soliq kodeksi",
    "sample_responses.txt": "Namunaviy javoblar",
    "bank_cases.txt": "Bank amaliyotlari",
    "bank_siri_lexuz.txt": "Bank siri (LEX.UZ)",
}

SOURCE_PRIORITY = {
    "bank_siri_lexuz.txt": 3.5,
    "bank_law.txt": 2.5,
    "bank_cases.txt": 2.0,
    "tax_code.txt": 1.6,
    "constitution.txt": 1.4,
    "sample_responses.txt": 1.0,
}

BANK_QUERY_TOPICS = {
    "bank_secrecy": (
        "bank siri",
        "hisobvaraq",
        "hisobvaraqlar",
        "tranzaksiya",
        "tranzaksiyalar",
        "karta",
        "kredit",
        "depozit",
        "mijoz",
        "bank ko'chirma",
        "bank ko‘chirma",
        "maxfiy",
        "oshkor",
    ),
    "prosecutor": (
        "prokuratura",
        "prokuror",
        "tergov",
        "surishtiruv",
        "tezkor-qidiruv",
    ),
    "court": (
        "sud",
        "ijro",
        "davlat ijrosi",
        "ijro ishi",
    ),
    "tax": (
        "soliq",
        "deklaratsiya",
        "hisobot",
        "soliq xizmati",
    ),
    "central_bank": (
        "markaziy bank",
        "mb",
        "regulyator",
        "nazorat",
    ),
}


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    source_filename: str
    source_title: str
    text: str
    article_refs: tuple[str, ...]

    @property
    def citation(self) -> str:
        article_part = "; ".join(f"{article}-modda" for article in self.article_refs)
        if article_part:
            return f"[{self.source_title}, {article_part}]"
        return f"[{self.source_title}]"


def split_words(text: str) -> list[str]:
    return [word for word in re.split(r"\s+", text.strip()) if word]


def chunk_text(
    text: str,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    words = split_words(text)
    if not words:
        return []

    if len(words) <= chunk_words:
        return [" ".join(words)]

    step = max(1, chunk_words - overlap_words)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def extract_article_refs(text: str) -> tuple[str, ...]:
    seen: list[str] = []
    for match in ARTICLE_PATTERN.finditer(text):
        article = match.group(1)
        if article not in seen:
            seen.append(article)
    return tuple(seen)


def build_legal_chunks(
    documents: Sequence[dict[str, str]],
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
) -> list[LegalChunk]:
    chunks: list[LegalChunk] = []

    for document in documents:
        filename = document["filename"]
        source_title = SOURCE_TITLE_OVERRIDES.get(filename, document["title"])
        content = document["content"]
        for index, chunk_text_value in enumerate(
            chunk_text(content, chunk_words=chunk_words, overlap_words=overlap_words)
        ):
            article_refs = extract_article_refs(chunk_text_value)
            chunk_id = f"{filename}::chunk-{index + 1}"
            chunks.append(
                LegalChunk(
                    chunk_id=chunk_id,
                    source_filename=filename,
                    source_title=source_title,
                    text=chunk_text_value,
                    article_refs=article_refs,
                )
            )
    return chunks


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def _detect_query_topics(query: str) -> set[str]:
    detected: set[str] = set()
    for topic, keywords in BANK_QUERY_TOPICS.items():
        if _contains_any(query, keywords):
            detected.add(topic)
    return detected


def _lexical_overlap_score(query: str, candidate_text: str) -> float:
    query_text = _normalize_text(query)
    candidate = _normalize_text(candidate_text)
    if not query_text or not candidate:
        return 0.0

    query_terms = [term for term in re.split(r"\s+", query_text) if term]
    if not query_terms:
        return 0.0

    matched = 0
    for term in query_terms:
        if len(term) < 3:
            continue
        if term in candidate:
            matched += 1
    return matched / max(1, len([term for term in query_terms if len(term) >= 3]))


def _source_priority(filename: str, query_topics: set[str]) -> float:
    priority = SOURCE_PRIORITY.get(filename, 1.0)

    if "bank_secrecy" in query_topics:
        if filename == "bank_siri_lexuz.txt":
            priority += 2.5
        elif filename == "bank_law.txt":
            priority += 1.8
        elif filename == "bank_cases.txt":
            priority += 1.3
        elif filename == "constitution.txt":
            priority += 0.6

    if "prosecutor" in query_topics:
        if filename == "bank_siri_lexuz.txt":
            priority += 1.5
        elif filename == "bank_cases.txt":
            priority += 1.0

    if "court" in query_topics:
        if filename == "bank_siri_lexuz.txt":
            priority += 1.4
        elif filename == "bank_cases.txt":
            priority += 0.9

    if "tax" in query_topics:
        if filename == "tax_code.txt":
            priority += 1.8
        elif filename == "bank_siri_lexuz.txt":
            priority += 0.6

    if "central_bank" in query_topics:
        if filename in {"bank_siri_lexuz.txt", "bank_law.txt"}:
            priority += 1.4

    return priority


class LegalRAGIndex:
    def __init__(
        self,
        chunks: Sequence[LegalChunk],
        *,
        vectorizer: TfidfVectorizer,
        faiss_index: faiss.Index,
        embeddings: np.ndarray,
    ) -> None:
        self.chunks = list(chunks)
        self.vectorizer = vectorizer
        self.faiss_index = faiss_index
        self.embeddings = embeddings

    @classmethod
    def build(
        cls,
        *,
        base_dir: Path | str = LEGAL_DOCS_DIR,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
        overlap_words: int = DEFAULT_CHUNK_OVERLAP,
    ) -> "LegalRAGIndex":
        ensure_knowledge_base(base_dir)
        documents = load_knowledge_base_documents(base_dir)
        chunks = build_legal_chunks(documents, chunk_words=chunk_words, overlap_words=overlap_words)
        if not chunks:
            empty_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
            empty_index = faiss.IndexFlatIP(MAX_FEATURES)
            empty_embeddings = np.zeros((0, MAX_FEATURES), dtype=np.float32)
            return cls([], vectorizer=empty_vectorizer, faiss_index=empty_index, embeddings=empty_embeddings)

        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            lowercase=True,
            ngram_range=(1, 2),
            stop_words=None,
        )
        matrix = vectorizer.fit_transform(chunk.text for chunk in chunks)
        embeddings = matrix.toarray().astype(np.float32)
        embeddings = normalize_matrix(embeddings)

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings)

        return cls(chunks, vectorizer=vectorizer, faiss_index=faiss_index, embeddings=embeddings)

    def _transform_query(self, query: str) -> np.ndarray:
        vector = self.vectorizer.transform([query]).toarray().astype(np.float32)
        vector = normalize_matrix(vector)
        return vector

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not self.chunks:
            return []

        query_vector = self._transform_query(query)
        candidate_limit = min(len(self.chunks), max(1, top_k * 6))
        scores, indices = self.faiss_index.search(query_vector, candidate_limit)

        query_topics = _detect_query_topics(query)
        reranked_results: list[dict[str, Any]] = []
        normalized_query = _normalize_text(query)

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            chunk = self.chunks[int(index)]
            lexical_overlap = _lexical_overlap_score(query, chunk.text)
            title_overlap = _lexical_overlap_score(query, chunk.source_title)
            source_priority = min(_source_priority(chunk.source_filename, query_topics) / 5.0, 1.0)
            exact_match_boost = 1.0 if normalized_query and normalized_query in _normalize_text(chunk.text) else 0.0

            final_score = (
                float(score) * 0.58
                + lexical_overlap * 0.22
                + title_overlap * 0.08
                + source_priority * 0.09
                + exact_match_boost * 0.03
            )

            reranked_results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_filename": chunk.source_filename,
                    "source_title": chunk.source_title,
                    "article_refs": list(chunk.article_refs),
                    "citation": chunk.citation,
                    "score": float(final_score),
                    "text": chunk.text,
                }
            )

        reranked_results.sort(key=lambda item: item["score"], reverse=True)
        return reranked_results[: min(max(1, top_k), len(reranked_results))]

    def build_context(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        retrieved = self.retrieve(query=query, top_k=top_k)
        if not retrieved:
            return ""

        sections: list[str] = []
        for item in retrieved:
            citation = item["citation"]
            score = item["score"]
            text = item["text"]
            sections.append(f"{citation} | moslik={score:.3f}\n{text}")
        return "\n\n".join(sections)


@lru_cache(maxsize=1)
def load_default_rag_index(base_dir: str = str(LEGAL_DOCS_DIR)) -> LegalRAGIndex:
    return LegalRAGIndex.build(base_dir=Path(base_dir))


def retrieve_legal_context(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    base_dir: Path | str = LEGAL_DOCS_DIR,
) -> list[dict[str, Any]]:
    index = load_default_rag_index(str(base_dir))
    return index.retrieve(query=query, top_k=top_k)


def build_legal_context_text(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    base_dir: Path | str = LEGAL_DOCS_DIR,
) -> str:
    index = load_default_rag_index(str(base_dir))
    return index.build_context(query=query, top_k=top_k)


def chunk_documents_from_knowledge_base(
    base_dir: Path | str = LEGAL_DOCS_DIR,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    ensure_knowledge_base(base_dir)
    documents = load_knowledge_base_documents(base_dir)
    chunks = build_legal_chunks(documents, chunk_words=chunk_words, overlap_words=overlap_words)
    return [
        {
            "chunk_id": chunk.chunk_id,
            "source_filename": chunk.source_filename,
            "source_title": chunk.source_title,
            "article_refs": list(chunk.article_refs),
            "citation": chunk.citation,
            "text": chunk.text,
        }
        for chunk in chunks
    ]


__all__ = [
    "DEFAULT_TOP_K",
    "LegalChunk",
    "LegalRAGIndex",
    "build_legal_context_text",
    "chunk_documents_from_knowledge_base",
    "load_default_rag_index",
    "retrieve_legal_context",
]
