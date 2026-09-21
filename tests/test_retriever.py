import os
import sys
import asyncio
from pathlib import Path

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
from app.retriever import HybridRetriever
from app.config import settings


async def run_retrieval_tests():
    print("\n=======================================================")
    print("HYBRID RETRIEVER TEST SUITE")
    print("=======================================================")

    retriever = HybridRetriever(app_settings=settings)
    retriever.load_indexes()

    print(f"Loaded {len(retriever.chunk_metadata)} chunks into memory.")
    print(f"BM25 corpus size: {retriever.bm25_index.corpus_size}")
    print(f"Dense embeddings shape: {retriever.embeddings.shape}")
    print("=======================================================\n")

    test_queries = [
        ("TEST 1", "What is the cancellation policy?"),
        ("TEST 2", "What are the hotel reimbursement limits?"),
        ("TEST 3", "What are the rules for international travel?"),
        ("TEST 4", "What are the catering requirements for an event?"),
        ("TEST 5", "international travel reimbursement and hotel booking"),
    ]

    for label, query in test_queries:
        print(f"-------------------------------------------------------")
        print(f"{label}: \"{query}\"")
        print(f"-------------------------------------------------------")

        # 1. BM25 results
        bm25_res = await retriever.retrieve_bm25_for_sub_query(query, top_k=5)
        print("BM25 Top-5:")
        for rank, (cid, score) in enumerate(bm25_res, start=1):
            meta = retriever.chunk_by_id[cid]
            sec = meta.get("section_title") or meta.get("section")
            print(f"  [{rank}] {cid} (score: {score:.4f}) | Doc: {meta['doc_id']} | Sec: {sec}")

        # 2. Dense results
        dense_res = await retriever.retrieve_dense_for_sub_query(query, top_k=5)
        print("Dense Top-5:")
        for rank, (cid, score) in enumerate(dense_res, start=1):
            meta = retriever.chunk_by_id[cid]
            sec = meta.get("section_title") or meta.get("section")
            print(f"  [{rank}] {cid} (similarity: {score:.4f}) | Doc: {meta['doc_id']} | Sec: {sec}")

        # 3. RRF Hybrid fusion
        rrf_res = retriever.compute_rrf(bm25_res, dense_res, k=settings.rrf_k)
        print("RRF Fusion Top-4:")
        for rank, (cid, score) in enumerate(rrf_res[:4], start=1):
            meta = retriever.chunk_by_id[cid]
            sec = meta.get("section_title") or meta.get("section")
            print(f"  [{rank}] {cid} (RRF score: {score:.6f}) | Doc: {meta['doc_id']} | Sec: {sec}")

        # 4. Multi-intent / sub-query method test
        evidence_list = await retriever.retrieve_for_sub_queries([query], session_id="test_session")
        print(f"Final Evidence Count: {len(evidence_list)}")
        print()

    # Test Multi-Subquery Decomposition Fusion with Minimum Evidence Guarantee
    print("=======================================================")
    print("MULTI-INTENT DECOMPOSED SUB-QUERY TEST")
    print("=======================================================")
    multi_sub_queries = [
        "What are the hotel nightly rate limits and booking rules?",
        "What happens if I cancel the event and venue contract?"
    ]
    print(f"Sub-queries: {multi_sub_queries}")
    multi_evidence = await retriever.retrieve_for_sub_queries(multi_sub_queries, session_id="multi_test")
    print(f"\nFinal Deduplicated & Guaranteed Evidence ({len(multi_evidence)} chunks):")
    for rank, ev in enumerate(multi_evidence, start=1):
        meta = retriever.chunk_by_id[ev.chunk_id]
        print(f"  [{rank}] Chunk: {ev.chunk_id} | SubQuery ID: {ev.sub_query_id} | RRF Score: {ev.score:.6f}")
        print(f"      Citation: [DOC: {ev.doc_id} | Section: {ev.section_title} | Chunk: {ev.chunk_id}]")
        print(f"      Snippet: {ev.text[:120]}...")
    print("=======================================================\n")


@pytest.mark.asyncio
async def test_retriever_pipeline():
    retriever = HybridRetriever(app_settings=settings)
    retriever.load_indexes()

    assert len(retriever.chunk_metadata) == 40
    assert retriever.bm25_index.corpus_size == 40
    assert retriever.embeddings.shape == (40, 768)

    # Test BM25
    bm25_res = await retriever.retrieve_bm25_for_sub_query("cancellation policy", top_k=4)
    assert len(bm25_res) > 0

    # Test Dense
    dense_res = await retriever.retrieve_dense_for_sub_query("cancellation policy", top_k=4)
    assert len(dense_res) > 0

    # Test RRF
    rrf_res = retriever.compute_rrf(bm25_res, dense_res)
    assert len(rrf_res) > 0

    # Test multi-subquery
    evidence = await retriever.retrieve_for_sub_queries(["hotel policy", "catering limit"])
    assert len(evidence) >= 2

    # Regression Test 1: Index Manifest validation
    assert retriever.index_manifest is not None, "Index manifest was not loaded."
    assert "embedding_backend" in retriever.index_manifest
    assert retriever.index_manifest["embedding_dimension"] == 768

    # Regression Test 2: Zero-score BM25 results must not be returned or injected into RRF
    zero_match_query = "xyzqwerty nonesuchterm 999999"
    bm25_zero = await retriever.retrieve_bm25_for_sub_query(zero_match_query, top_k=4)
    assert len(bm25_zero) == 0, f"Expected 0 BM25 results for unmatched query, got {len(bm25_zero)}"

    # Dense results should still work independently
    dense_for_zero = await retriever.retrieve_dense_for_sub_query(zero_match_query, top_k=4)
    assert len(dense_for_zero) > 0

    # RRF should only contain dense candidates without zero-score BM25 chunks
    rrf_zero = retriever.compute_rrf(bm25_zero, dense_for_zero)
    assert len(rrf_zero) == len(dense_for_zero)
    # The top chunk in RRF should match the top dense chunk
    assert rrf_zero[0][0] == dense_for_zero[0][0]


if __name__ == "__main__":
    asyncio.run(run_retrieval_tests())
