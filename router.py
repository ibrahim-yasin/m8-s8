"""Module 8 — Tuesday Stretch (Honors Track): Query Router.

Build a routing layer that classifies an incoming query into one of three
types and dispatches it to a different retrieval pipeline:
    factoid (rare entity, date, figure) -> BM25
    semantic (paraphrastic, descriptive) -> dense
    mixed / unknown -> hybrid (alpha=0.5)
"""

from __future__ import annotations

import re
import weaviate

from retrieval_helpers import bm25_search, dense_search, hybrid_search


def classify_query(query: str) -> str:
    """Return one of 'factoid', 'semantic', 'mixed'."""
    q = query.strip()
    words = q.split()

    has_digits = bool(re.search(r"\d", q))
    has_quotes = bool(re.search(r'"[^"]+"', q))

    has_code_like = bool(
        re.search(
            r"(\b[A-Za-z_]*[A-Z][A-Za-z_]*[A-Z][A-Za-z0-9_]*\b|"
            r"\b[A-Za-z_]+\.[A-Za-z0-9_.]+\b|"
            r"\b0x[0-9a-fA-F]+\b|"
            r"\b[A-Z_]{3,}\b|"
            r"\b\w+\(\)|"
            r"\b\d+\.\d+(?:\.\d+)*\b|"
            r"[-_/])",
            q,
        )
    )

    factoid_keywords = [
        "what is",
        "what does",
        "how does",
        "where can i find",
        "why does",
        "what causes",
    ]

    paraphrastic_keywords = [
        "tips",
        "should",
        "can i",
        "can a",
        "how can i",
        "how to",
        "best",
        "improve",
        "drawback",
        "efficiently",
        "required",
        "really",
    ]

    factoid_signals = 0
    semantic_signals = 0

    if has_digits:
        factoid_signals += 2
    if has_quotes:
        factoid_signals += 2
    if has_code_like:
        factoid_signals += 3

    q_lower = q.lower()

    if any(k in q_lower for k in factoid_keywords):
        factoid_signals += 1

    if any(k in q_lower for k in paraphrastic_keywords):
        semantic_signals += 2

    if len(words) <= 5:
        semantic_signals += 1

    if len(words) > 12 and not has_code_like:
        semantic_signals += 1

    if factoid_signals >= semantic_signals + 2:
        return "factoid"

    if semantic_signals >= factoid_signals:
        return "semantic"

    return "mixed"


def routed_search(client: weaviate.Client, query: str, k: int, embedder) -> list[str]:
    """Dispatch to BM25 / dense / hybrid based on classify_query(query).

    Return the ordered list of doc_id strings, length <= k.
    """
    # TODO: kind = classify_query(query)
    # TODO: dispatch:
    #         "factoid"  -> bm25_search(client, query, k)
    #         "semantic" -> dense_search(client, query, k, embedder)
    #         else       -> hybrid_search(client, query, k, embedder, alpha=0.5)
    kind = classify_query(query)
    if kind == "factoid":
        return bm25_search(client, query, k)
    elif kind == "semantic":
        return dense_search(client, query, k, embedder)
    else:
        return hybrid_search(client, query, k, embedder, alpha=0.5)