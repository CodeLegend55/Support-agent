"""
retrieval.py
------------
Lightweight retrieval (RAG) over the policy markdown files.

No embedding API is used so this runs fully offline: documents are
chunked by section (## headers), turned into TF-IDF vectors with a
plain-Python implementation, and ranked by cosine similarity against
the query. Good enough for a handful of short policy docs; swap in a
real embedding model if the corpus grows.
"""

import glob
import math
import os
import re
from collections import Counter
from typing import Optional

POLICY_DIR = os.path.join(os.path.dirname(__file__), "data", "policies")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "for", "to", "and", "or", "with", "as", "at", "by",
    "from", "this", "that", "it", "its", "if", "do", "does", "did", "my",
    "your", "i", "you", "can", "will", "what", "how", "when", "which",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9%₹]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _load_chunks() -> list[dict]:
    """Split each markdown file into chunks by '## ' section headers."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.md"))):
        source = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()

        doc_title_match = re.match(r"#\s+(.+)", content)
        doc_title = doc_title_match.group(1).strip() if doc_title_match else source

        sections = re.split(r"\n(?=## )", content)
        for sec in sections:
            sec = sec.strip()
            if not sec or sec.startswith("# "):
                # the very first split piece is the "# Title" line only, skip if empty of content
                if sec.startswith("## "):
                    pass
                else:
                    continue
            header_match = re.match(r"##\s+(.+)", sec)
            section_title = header_match.group(1).strip() if header_match else doc_title
            chunks.append({
                "source": source,
                "doc_title": doc_title,
                "section": section_title,
                "text": sec,
            })
    return chunks


_CHUNKS = _load_chunks()


def _build_index(chunks: list[dict]):
    doc_tokens = [_tokenize(c["section"] + " " + c["text"]) for c in chunks]
    df = Counter()
    for toks in doc_tokens:
        for term in set(toks):
            df[term] += 1
    n_docs = len(chunks)
    idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1 for term, freq in df.items()}

    vectors = []
    for toks in doc_tokens:
        tf = Counter(toks)
        vec = {term: (count / len(toks)) * idf.get(term, 0.0) for term, count in tf.items()} if toks else {}
        vectors.append(vec)
    return vectors, idf


_VECTORS, _IDF = _build_index(_CHUNKS)


def _cosine(vec_a: dict, vec_b: dict) -> float:
    common = set(vec_a) & set(vec_b)
    num = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return num / (norm_a * norm_b)


def retrieve(query: str, top_k: int = 3, min_score: float = 0.05) -> list[dict]:
    """
    Return the top_k policy chunks most relevant to the query, each with
    source file, section title, text, and similarity score. Empty list
    if nothing clears min_score (i.e. the docs likely don't cover this).
    """
    q_tokens = _tokenize(query)
    tf = Counter(q_tokens)
    q_vec = {term: (count / len(q_tokens)) * _IDF.get(term, 0.0) for term, count in tf.items()} if q_tokens else {}

    scored = []
    for chunk, vec in zip(_CHUNKS, _VECTORS):
        score = _cosine(q_vec, vec)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, chunk in scored[:top_k]:
        if score < min_score:
            continue
        results.append({**chunk, "score": round(score, 4)})
    return results


if __name__ == "__main__":
    for r in retrieve("what's your return policy for electronics"):
        print(r["source"], r["section"], r["score"])
