# Streaming RAG Benchmark & Evaluation Report

This document records the empirical evaluation results and comparative benchmarks for the Streaming RAG system across architectural iterations and ablation configurations.

---

## 1. Evaluation Methodology & Datasets
<!-- TODO: Document dataset sources, test transcripts, and corpus size. -->
- **Test Corpus**: Raw documents in `corpus/raw_docs/` with section annotations.
- **Evaluation Transcripts**: Simulated multi-turn audio transcripts in `eval/test_transcripts/`.
- **Harness**: `eval/replay_harness.py` (simulating realtime audio streaming deltas).
- **Compliance Gates**: `eval/gate_check.py` evaluating Gates G1 through G6.

---

## 2. Retrieval Strategy Ablation Benchmarks

| Configuration | Recall@4 | Intent Coverage (%) | Retrieval Latency (ms) | Minimum Evidence Pass (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Dense-Only Global Top-K)** | *TODO* | *TODO* | *TODO* | *TODO* |
| **BM25 Sparse Only** | *TODO* | *TODO* | *TODO* | *TODO* |
| **Global Hybrid (BM25 + Dense RRF)** | *TODO* | *TODO* | *TODO* | *TODO* |
| **Multi-Intent Per-Query RRF (No Rerank)**| *TODO* | *TODO* | *TODO* | *TODO* |
| **Multi-Intent Per-Query RRF + Rerank (Full)** | *TODO* | *TODO* | *TODO* | *TODO* |

### Key Findings & Observations
<!-- TODO: Record observations on why per-sub-query RRF prevents intent starvation. -->

---

## 3. End-to-End Latency Breakdown

| Pipeline Stage | Mean Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **T0 Stability Gate** | *TODO* | *TODO* | *TODO* | Lexical / embedding delta |
| **T1 LLM Decomposition** | *TODO* | *TODO* | *TODO* | Structured routing |
| **Hybrid Retrieval (Per-Query)** | *TODO* | *TODO* | *TODO* | BM25 + Dense + RRF |
| **Cross-Encoder Reranking** | *TODO* | *TODO* | *TODO* | Applied on candidate union |
| **Claim Synthesis** | *TODO* | *TODO* | *TODO* | First claim generation |
| **Cheap Local Grounding** | *TODO* | *TODO* | *TODO* | Fast heuristic pass |
| **Batched LLM Verification** | *TODO* | *TODO* | *TODO* | Fallback on borderline claims |
| **Time to First Verified Claim (TTFC)** | *TODO* | *TODO* | *TODO* | User perception metric |

---

## 4. Factual Grounding & Verification Metrics

| Metric | Target | Measured Value | Gate Status |
| :--- | :--- | :--- | :--- |
| **Claim Precision (Grounding Accuracy)** | $\ge 98\%$ | *TODO* | G4 - *TODO* |
| **Hallucinated Claims Streamed** | $0\%$ | *TODO* | G4 - *TODO* |
| **Partially Supported Softening Accuracy** | $\ge 90\%$ | *TODO* | G4 - *TODO* |
| **Uncertainty Section Recall** | $\ge 95\%$ | *TODO* | G4 - *TODO* |

---

## 5. Multi-Intent & Refinement Performance

| Scenario | Intent Recall (%) | Redundant Retrieval Rate (%) | Answer Version Consistency |
| :--- | :--- | :--- | :--- |
| **Single Turn Multi-Intent (3 intents)** | *TODO* | *N/A* | *TODO* |
| **Refinement Turn (Delta constraint)** | *TODO* | *TODO (Target < 5%)* | *TODO* |
| **No-Retrieval Turn (Reformatting)** | *N/A* | *0% (Bypass)* | *TODO* |

---

## 6. Architectural Gate Compliance (G1 - G6)

- [ ] **G1 - Reproducibility**: Deterministic index construction and ranking stability.
- [ ] **G2 - Early Retrieval**: Sub-utterance retrieval trigger upon T0 intent stability.
- [ ] **G3 - Multi-Intent Identification**: Full sub-query coverage and minimum evidence guarantee ($\ge 2$ chunks/intent).
- [ ] **G4 - Factual Grounding**: 100% of streamed claims pass verification; citations ground accurately to section level.
- [ ] **G5 - Session Refinement**: State preserved across turns; zero redundant retrievals for covered intents.
- [ ] **G6 - Telemetry & Observability**: Complete JSONL telemetry records with latencies, scores, and token counts.
