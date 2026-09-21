# Streaming RAG Architecture Brief

## 1. Problem Statement
<!-- TODO: Detail the challenge of low-latency, factual retrieval-augmented generation over continuous real-time streaming transcripts (e.g. meetings, voice calls, live broadcasts). -->

- Standard batch RAG waits for a complete user utterance before retrieving and generating, leading to unacceptable latency in live voice/conversational contexts.
- Naive streaming RAG streams unverified LLM output token-by-token, risking hallucination of ungrounded statements before verification can intervene.
- Multi-intent queries often result in "intent starvation", where dominant search terms crowd out subtle constraints.

---

## 2. System Architecture & High-Level Flow

```text
TRANSCRIPT STREAM
        ↓
ASYNC STREAM QUEUE
        ↓
2-TIER CONTROLLER
        ↓
retrieve?
   ┌────┴────┐
   │         │
 FALSE      TRUE
   │         │
   │         ↓
   │    MULTI-INTENT DECOMPOSITION
   │         ↓
   │    FOR EACH SUB-QUERY:
   │         ↓
   │    BM25 + Dense Retrieval
   │         ↓
   │    Per-sub-query RRF
   │         ↓
   │    Top-4 per sub-query
   │         ↓
   │    Union + Deduplicate
   │         ↓
   │    Final Reranker
   │         ↓
   │    Minimum evidence guarantee
   │         ↓
   └────→ SESSION STATE
              ↓
          SYNTHESIS
              ↓
       STRUCTURED CLAIMS
              ↓
       CLAIM-LEVEL VERIFIER
              ↓
       VERIFIED CLAIM
              ↓
       STREAM TO USER
              ↓
    ANSWER + CITATIONS + UNCERTAINTY
```

---

## 3. Two-Tier Controller (T0 & T1)
<!-- TODO: Document controller implementation specifications. -->

### Tier 0: Intent Stability Gate
- Lightweight lexical/embedding-delta check.
- Prevents triggering expensive LLM calls on partial or unstable speech transcript syllables.

### Tier 1: Multi-Intent Router & Decomposition
- Single structured LLM call.
- Evaluates:
  - `retrieve`: boolean (True for information lookup, False for formatting/conversational turns).
  - `is_refine`: boolean (True if modifying/expanding a prior answer).
  - `sub_queries`: list of atomic decomposed search queries.
- **Critical Rule**: T1 MUST read `SessionState` (covered intents, previous claims, previous answer) before making decisions so that refinements search ONLY for missing/new intents.

---

## 4. Multi-Intent Hybrid Retrieval Strategy
<!-- TODO: Document hybrid retrieval, fusion, and reranking parameters. -->

1. **Independent Per-Sub-Query Retrieval**:
   - Each sub-query independently executes BM25 sparse search and dense vector search.
   - Per-sub-query Reciprocal Rank Fusion (RRF with $k=60$) isolates intents and produces Top-4 candidate chunks per sub-query.
   - *Why*: Prevents global competition where a dominant general intent starves a specific intent.
2. **Candidate Aggregation**:
   - Union all per-sub-query candidate sets.
   - Deduplicate identical chunk IDs.
3. **Cross-Encoder Reranking & Minimum Evidence Guarantee**:
   - Cross-encoder reranks the deduplicated candidate set.
   - Enforce Minimum Evidence Guarantee: Every sub-query MUST retain at least its top 2 chunks in the final evidence pool.

---

## 5. Session State & Refinement
<!-- TODO: Document session persistence, evidence caching, and refinement mechanisms. -->

- Persistent memory per session tracks:
  - `evidence_pool`: Deduplicated retrieved chunks with provenance metadata.
  - `claims`: List of structured claims with verification status.
  - `covered_intents`: Set of satisfied user intents.
  - `answer_version`: Integer tracking incremental response revisions.
- **Refinement Rule**: Do NOT clear state or re-retrieve covered intents. Search only for delta constraints and update structured claims incrementally.

---

## 6. Structured Claim Synthesis & Claim-Level Grounding
<!-- TODO: Document claim schema, tiered verification heuristics, and streaming order. -->

### Structured Claims
- Synthesis outputs discrete claim objects:
  ```json
  {
    "claims": [
      {
        "id": 1,
        "text": "Venue A accommodates 30 people",
        "cites": ["Doc_12 §2"],
        "intent": 0
      }
    ]
  }
  ```

### Two-Step Verification Strategy
1. **Step 1 (Cheap Local Check)**: Entity overlap, lexical matching, and embedding similarity. Unambiguously supported claims pass immediately.
2. **Step 2 (Batched LLM Verification)**: Borderline claims are batched into a single LLM verification call.

### Verified Streaming Order
- Flow: `Generate Claim 1 -> Verify Claim 1 -> Stream Claim 1`
- Unsupported claims are removed from the main answer and diverted to the Uncertainty section.

---

## 7. No-Retrieval Path
<!-- TODO: Document bypass conditions for conversational or formatting turns. -->

- When `retrieve=False` (e.g. "format previous answer as bullet points"):
  - Bypasses BM25/dense/reranker corpus operations.
  - Synthesizes directly from `SessionState.claims` and `SessionState.previous_answer`.

---

## 8. Telemetry & Observability Side-Channel
<!-- TODO: Document telemetry event schemas and JSONL logging. -->

- Non-blocking structured logging writes all events to `logs/session_<session_id>.jsonl`.
- Captures: latency breakdown, T0/T1 decisions, per-chunk scores, RRF ranks, reranker scores, claim verifications, citations, and token costs.

---

## 9. Evaluation Framework & Quality Gates
<!-- TODO: Document evaluation harness and compliance criteria (G1-G6). -->

- **G1 (Reproducibility)**: Deterministic retrieval and indexing.
- **G2 (Early Retrieval)**: Timely retrieval trigger relative to transcript intent stability.
- **G3 (Multi-Intent Identification)**: Accurate decomposition and minimum evidence retention.
- **G4 (Factual Grounding)**: Zero unverified claims in streamed output; accurate citation links.
- **G5 (Session Refinement)**: Incremental updates without re-retrieval of covered intents.
- **G6 (Telemetry Completeness)**: Comprehensive event logging and lineage.
