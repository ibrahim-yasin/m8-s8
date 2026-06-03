import json
from collections import defaultdict
from typing import Callable
from router import routed_search, classify_query
from router import routed_search
from retrieval_helpers import bm25_search, dense_search, hybrid_search


def evaluate_retriever(eval_path: str, search_fn: Callable, k_values=(5, 10)) -> dict:
    max_k = max(k_values)

    def empty_stats():
        return {
            "count": 0,
            "hits": {k: 0 for k in k_values},
            "mrr_sum": 0.0,
        }

    overall = empty_stats()
    by_type_stats = defaultdict(empty_stats)

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            query = row["query"]
            gold_doc_id = row["gold_doc_id"]
            query_type = row["query_type"]

            results = search_fn(query, k=max_k)

            returned_ids = []
            for r in results:
                if isinstance(r, str):
                    returned_ids.append(r)
                elif isinstance(r, dict):
                    returned_ids.append(r.get("doc_id") or r.get("id"))
                else:
                    returned_ids.append(
                        getattr(r, "doc_id", None) or getattr(r, "id", None)
                    )

            overall["count"] += 1
            by_type_stats[query_type]["count"] += 1

            for k in k_values:
                if gold_doc_id in returned_ids[:k]:
                    overall["hits"][k] += 1
                    by_type_stats[query_type]["hits"][k] += 1

            mrr = 0.0
            top_ids = returned_ids[:max_k]

            if gold_doc_id in top_ids:
                rank = top_ids.index(gold_doc_id) + 1
                mrr = 1.0 / rank

            overall["mrr_sum"] += mrr
            by_type_stats[query_type]["mrr_sum"] += mrr

    def finalize(stats):
        n = stats["count"]

        if n == 0:
            return {
                **{f"recall@{k}": 0.0 for k in k_values},
                "mrr": 0.0,
            }

        return {
            **{f"recall@{k}": stats["hits"][k] / n for k in k_values},
            "mrr": stats["mrr_sum"] / n,
        }

    return {
        **finalize(overall),
        "by_type": {
            query_type: finalize(stats)
            for query_type, stats in by_type_stats.items()
        },
    }

def evaluate_classifier_accuracy(eval_path: str, classify_fn: Callable, n: int = 20) -> dict:
    correct = 0
    total = 0

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            query = row["query"]
            gold = row["query_type"]

            pred = classify_fn(query)

            
            if pred == "semantic":
                pred = "paraphrastic"

            if pred == gold:
                correct += 1

            total += 1

            if total >= n:
                break

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


if __name__ == "__main__":
    import weaviate
    from sentence_transformers import SentenceTransformer

    client = weaviate.Client("http://localhost:8080")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("\n=== BM25 ===")
    print(
        evaluate_retriever(
            "data/retrieval_eval.jsonl",
            lambda q, k: bm25_search(client, q, k),
        )
    )

    print("\n=== DENSE ===")
    print(
        evaluate_retriever(
            "data/retrieval_eval.jsonl",
            lambda q, k: dense_search(client, q, k, embedder),
        )
    )

    print("\n=== HYBRID ===")
    print(
        evaluate_retriever(
            "data/retrieval_eval.jsonl",
            lambda q, k: hybrid_search(client, q, k, embedder, alpha=0.5),
        )
    )

    print("\n=== ROUTED ===")
    print(
        evaluate_retriever(
            "data/retrieval_eval.jsonl",
            lambda q, k: routed_search(client, q, k, embedder),
        )
    )

    print("\n=== CLASSIFIER ACCURACY ===")
    print(
        evaluate_classifier_accuracy(
            "data/retrieval_eval.jsonl",
            classify_query,
            n=20,
        )
    )
