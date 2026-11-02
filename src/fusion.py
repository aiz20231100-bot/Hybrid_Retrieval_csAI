def rrf_fuse(rankings, k=60, top_k=10):
    """rankings: قائمة من القوائم، كل قائمة [(doc_id, score), ...] مرتبة تنازلياً."""
    scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]