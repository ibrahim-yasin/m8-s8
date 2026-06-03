# Routing Report — Module 8 Tuesday Stretch

## 1. Per-Query-Type Classifier Accuracy

I implemented a rule-based query classifier that assigns queries to one of three categories: factoid, semantic, or mixed. The classifier uses simple heuristics based on query characteristics. Queries containing exact identifiers, numbers, version strings, error codes, API names, or highly specific technical terms are classified as factoid. Queries that appear paraphrastic, descriptive, or focused on semantic meaning rather than exact lexical overlap are classified as semantic. Queries containing signals from both categories are classified as mixed.

The motivation behind this design is that BM25 generally performs best on exact-match factoid queries, while dense retrieval performs better on semantic and paraphrastic queries. The router attempts to exploit these strengths by selecting the retrieval strategy most likely to succeed for each query type.

The classifier was evaluated on a held-out subset of 20 labeled queries and achieved an accuracy of **95% (19/20 correct predictions)**. This indicates that the heuristic rules were highly effective at distinguishing factoid queries from semantic and paraphrastic queries. The single observed error occurred on a query whose lexical and semantic characteristics overlapped, making classification more ambiguous.

---

## 2. Routed Retriever Metrics

| Retriever               | recall@5 | recall@10 |   MRR |
| ----------------------- | -------: | --------: | ----: |
| BM25 (baseline)         |    0.567 |     0.617 | 0.536 |
| Dense (baseline)        |    0.900 |     0.933 | 0.670 |
| Hybrid α=0.5 (baseline) |    0.833 |     0.983 | 0.682 |
| Routed                  |    0.900 |     0.950 | 0.695 |

The results show clear differences between retrieval strategies. BM25 performed exceptionally well on factoid queries, achieving recall@5 of 1.000 and MRR of 0.967, but struggled on paraphrastic queries where recall@5 dropped to 0.133. Dense retrieval showed the opposite behavior, achieving recall@5 of 0.967 on paraphrastic queries while maintaining strong performance on factoid questions. Hybrid retrieval achieved the highest recall@10 (0.983), providing the broadest coverage of relevant documents.

The routed system achieved recall@5 of **0.900**, recall@10 of **0.950**, and MRR of **0.695**. It matched the Dense retriever on recall@5, exceeded both Dense and Hybrid retrieval on MRR, and substantially outperformed BM25 across all metrics. These results demonstrate that effective query classification can successfully route queries to the retrieval strategy most likely to return highly ranked relevant documents.

---

## 3. When Does Routing Win, When Does It Lose, Why

Routing performed best on clearly identifiable factoid queries containing unique lexical identifiers such as “0x55555555”, “ENIAC”, and “EXC_BAD_ACCESS”, where BM25 could exploit exact keyword matching to retrieve the correct document efficiently. Factoid queries benefited from routing because the classifier reliably detected technical identifiers, numeric values, and code-like patterns that are strong indicators of keyword-sensitive retrieval.

For semantic and paraphrastic queries, dense retrieval was generally superior. Queries such as “Getting Overwhelmed: Tips for noobs”, “Turn away a bug if no reproducible test case exists?”, and “Can a developer perform testing efficiently?” contain little lexical overlap with their relevant documents, making semantic similarity much more important than exact term matching. Routing successfully identified many of these queries and dispatched them to dense retrieval, improving ranking quality.

The routed system matched Dense retrieval on recall@5 (0.900) and achieved the highest MRR (0.695), indicating that it frequently ranked the correct document closer to the top of the retrieved list. However, Hybrid retrieval still achieved the highest recall@10 (0.983), suggesting that combining sparse and dense signals remains beneficial for maximizing overall document coverage.

Routing failures occurred primarily when a query contained both lexical and semantic characteristics. These mixed queries can benefit from Hybrid retrieval because they require both exact keyword matching and semantic understanding. Future improvements could include replacing the rule-based classifier with an embedding-based classifier, introducing confidence-based fallback to Hybrid retrieval, or learning routing decisions directly from retrieval performance data. Such improvements would likely reduce routing errors and further improve retrieval effectiveness.
