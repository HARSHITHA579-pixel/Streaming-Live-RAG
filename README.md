# Streaming RAG

A low-latency, factually grounded **Live Streaming Retrieval-Augmented Generation (RAG)** system designed for continuous transcript streams, multi-intent reasoning, and verified claim streaming.

---

## Architecture Overview

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

### Key Architectural Pillars

1. **Two-Tier Controller**:
   - **T0**: Intent stability gate using embedding delta to avoid invoking expensive LLMs on partial utterances.
   - **T1**: Reads persistent `SessionState` and produces structured decisions: `retrieve`, `is_refine`, and decomposed `sub_queries`. Refinement searches only for delta intents.
   - **No-Retrieval Bypass**: Directly synthesizes formatting or conversational turns from existing state without querying the corpus.
2. **Multi-Intent Hybrid Retrieval & RRF Isolation**:
   - Independent BM25 and dense retrieval per sub-query fused via per-query Reciprocal Rank Fusion (RRF) to extract Top-4 candidates.
   - Candidate union and deduplication followed by cross-encoder reranking.
   - **Minimum Evidence Guarantee**: Retains at least 2 chunks per sub-query in final evidence set to prevent intent starvation.
3. **Structured Claim Synthesis**:
   - Synthesizes discrete, citeable claim objects (`id`, `text`, `cites`, `intent`) rather than unconstrained raw paragraphs.
4. **Claim-Level Grounding & Verified Streaming**:
   - Tiered verification: Fast local heuristic checks for clear matches, with borderline claims sent in single batched LLM verification calls.
   - Pipelined streaming: `Claim 1 -> Verify Claim 1 -> Stream Claim 1` ensures no hallucinated claim is exposed before verification.
   - Unsupported claims are redirected to the Uncertainty section.
5. **Observability Side-Channel**:
   - All pipeline components log structured JSONL telemetry to `logs/session_<id>.jsonl`.

---

## Project Structure

```text
streaming-rag/
│
├── docker-compose.yml              # Container service definition
├── Dockerfile                      # Python container build
├── requirements.txt                # Project dependencies
├── .env.example                    # Environment variable template
├── README.md                       # Project overview and instructions
│
├── corpus/
│   ├── raw_docs/                   # Source documents for knowledge base
│   │   └── .gitkeep
│   │
│   └── build_index.py              # Parses docs, generates BM25 index & dense embeddings
│
├── data/                           # Serialized indices, embeddings, and metadata mapping
│   └── .gitkeep
│
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI server, WebSocket endpoint, pipeline orchestrator
│   ├── config.py                   # Pydantic configuration & environment settings
│   ├── controller.py               # 2-Tier Controller (T0 stability gate, T1 multi-intent router)
│   ├── retriever.py                # Per-sub-query BM25 + Dense + RRF + Reranker + Min Guarantee
│   ├── session_state.py            # Persistent conversation memory & evidence pool
│   ├── synthesis.py                # Structured claim generation & streaming
│   ├── verifier.py                 # Claim-level grounding (cheap local check + batched LLM)
│   └── telemetry.py                # Side-channel structured JSONL logging
│
├── logs/                           # Session telemetry JSONL files
│   └── .gitkeep
│
├── dashboard/
│   └── timeline.html               # Interactive visual telemetry & execution timeline dashboard
│
├── eval/
│   ├── replay_harness.py           # Replays timed streaming transcripts to simulate live feed
│   ├── gate_check.py               # Evaluates architectural quality gates G1 through G6
│   └── test_transcripts/           # Benchmark transcript test files
│       └── .gitkeep
│
├── tests/                          # Unit and integration tests
│   └── .gitkeep
│
└── docs/
    ├── architecture_brief.md       # Detailed technical design specifications
    └── benchmark_report.md         # Empirical metrics, ablation results, and gate scorecard
```

---

## Setup & Installation

### 1. Environment Setup

```bash
# Clone or navigate to the repository
cd /path/to/streaming-rag

# Create and activate a Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies (TODO: Install after components are ready to test)
# pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and supply your LLM API keys and model choices
```

---

## How to Build the Corpus Index

> **Status**: *[TODO - Scaffold created in `corpus/build_index.py`]*

1. Place raw text or markdown documents in `corpus/raw_docs/`.
2. Run the indexing script:

```bash
# [TODO: Implement corpus indexing before running]
python corpus/build_index.py
```

Expected outputs generated in `data/`:
- `data/bm25_index.pkl` (BM25 token index)
- `data/embeddings.npy` (Dense embedding matrix)
- `data/chunk_metadata.json` (Provenance mapping for chunk IDs, document IDs, section titles)

---

## How to Run the Application

> **Status**: *[TODO - FastAPI server scaffold created in `app/main.py`]*

### Running Locally with Uvicorn

```bash
# [TODO: Implement pipeline components before running server]
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Running with Docker

```bash
# [TODO: Build and run containerized service]
docker compose up --build
```

### Connecting to Live Streaming WebSocket

Connect to:
`ws://localhost:8000/ws/stream/{session_id}`

Send streaming text / transcript chunks and receive incremental verified claims, citations, uncertainty notes, and answer versions.

---

## How to Run Evaluation & Benchmarks

> **Status**: *[TODO - Harness created in `eval/replay_harness.py` and `eval/gate_check.py`]*

### Replaying Timed Transcripts

```bash
# [TODO: Place transcript files in eval/test_transcripts/ and run replay]
python eval/replay_harness.py
```

### Verifying Architectural Gates (G1 - G6)

```bash
# [TODO: Run automated compliance gate checks against session telemetry]
python eval/gate_check.py
```

### Viewing Execution Timeline Dashboard

Open `dashboard/timeline.html` in any web browser and load a session log from `logs/session_<session_id>.jsonl` to inspect the retrieval timeline, claim verification states, and latency breakdown.
