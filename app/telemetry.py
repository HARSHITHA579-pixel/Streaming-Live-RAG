"""
Structured Telemetry Logger for Streaming RAG

================================================================================
RESPONSIBILITY:
- Provide a standardized, non-blocking structured logging interface (side-channel).
- Log granular telemetry events across every stage of the pipeline into session-specific
  JSONL log files (`logs/session_<session_id>.jsonl`).
- Capture timing metrics, retrieval decisions, sub-query decomposition, per-chunk rankings/scores,
  claim verification results, answer version lineage, and token costs.

INPUTS:
- Telemetry event records emitted from Controller, Retriever, Verifier, Synthesis, and Server.

OUTPUTS:
- Machine-readable JSONL log files in the configured `logs/` directory.

CONNECTED COMPONENTS:
- `app.controller`: Logs T0 stability deltas, T1 decision, is_refine flag, and sub-queries.
- `app.retriever`: Logs per-sub-query BM25/dense scores, RRF ranks, union/dedup stats, and reranker scores.
- `app.verifier`: Logs local check outcomes, LLM batched verification results, and claim status transitions.
- `app.synthesis`: Logs claim generation latency, claims count, and citation metrics.
- `app.main`: Logs end-to-end request latencies, retrieval summaries, and stream lifecycle.
- `dashboard/timeline.html`: Consumes the generated JSONL log files for visual execution inspection.
- `eval/gate_check.py`: Parses telemetry JSONL to verify gate metrics (G1-G6).

WHY THIS ARCHITECTURE:
- Telemetry is a side-channel, NOT a sequential pipeline stage.
- Machine-readable JSON Lines format allows post-hoc replay, latency breakdown,
  and automated compliance validation without interfering with streaming performance.
================================================================================
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class TelemetryEvent:
    """
    Schema for individual telemetry events across the streaming RAG pipeline.
    """
    session_id: str
    event: str  # e.g., "T0_GATE", "T1_DECISION", "RETRIEVAL", "RETRIEVAL_BM25", "RETRIEVAL_DENSE", "RETRIEVAL_RRF", "RETRIEVAL_CHUNK", "SYNTHESIS", "CLAIM_VERIFY", "E2E"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: Optional[float] = None
    answer_version: Optional[int] = None

    # Decision / Action
    decision_action: Optional[str] = None  # "WAIT", "RETRIEVE", "NO_RETRIEVAL", "REFINE"

    # Controller details (T0 / T1)
    is_stable: Optional[bool] = None
    delta_similarity: Optional[float] = None
    should_retrieve: Optional[bool] = None
    retrieval_decision: Optional[bool] = None  # Alias for should_retrieve for backwards compatibility
    is_refine: Optional[bool] = None
    sub_queries: Optional[List[str]] = None
    reasoning: Optional[str] = None

    # Retrieval details (RETRIEVAL / RETRIEVAL_BM25 / RETRIEVAL_DENSE / RETRIEVAL_RRF / RETRIEVAL_CHUNK)
    number_of_subqueries: Optional[int] = None
    evidence_count: Optional[int] = None
    retrieval_call_count: Optional[int] = None
    sub_query_id: Optional[int] = None
    query: Optional[str] = None
    result_count: Optional[int] = None
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    section: Optional[str] = None
    retrieval_method: Optional[str] = None  # "bm25", "dense", "rrf", "rerank"
    rank: Optional[int] = None
    score: Optional[float] = None
    reranker_score: Optional[float] = None

    # Synthesis & Grounding details (SYNTHESIS)
    claims_generated: Optional[int] = None
    citations_generated: Optional[int] = None
    is_refinement: Optional[bool] = None

    # Verification details (CLAIM_VERIFY)
    claim_id: Optional[int] = None
    claim_text: Optional[str] = None
    claim_verification_result: Optional[str] = None  # "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"
    verification_method: Optional[str] = None  # "cheap_local", "batched_llm", "local", "llm"
    citation_count: Optional[int] = None
    citations: Optional[List[str]] = None

    # E2E details (E2E)
    total_latency_ms: Optional[float] = None
    controller_latency_ms: Optional[float] = None
    retrieval_latency_ms: Optional[float] = None
    synthesis_latency_ms: Optional[float] = None
    verification_latency_ms: Optional[float] = None

    # Token & cost estimates
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    total_tokens: Optional[int] = None
    model: Optional[str] = None
    usage_available: Optional[bool] = None
    estimated_cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def event_type(self) -> str:
        """Alias for backwards compatibility with earlier scaffold references."""
        return self.event

    @event_type.setter
    def event_type(self, value: str) -> None:
        self.event = value


class TelemetryLogger:
    """
    Asynchronous / thread-safe logger for writing telemetry events to session-isolated JSONL files.
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the telemetry logger with the destination directory.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get_session_log_path(self, session_id: str) -> Path:
        """Returns the file path for a session's telemetry JSONL log."""
        clean_id = session_id if session_id and session_id.strip() else "default"
        direct_path = self.log_dir / f"session_{clean_id}.jsonl"
        if not direct_path.exists() and clean_id.startswith("session_"):
            alt_path = self.log_dir / f"{clean_id}.jsonl"
            if alt_path.exists():
                return alt_path
        return direct_path

    def log_event(self, event: TelemetryEvent) -> None:
        """
        Appends a single telemetry event as a JSON line to the session log file.
        Ensures thread-safe write and valid JSON formatting.
        """
        try:
            log_path = self.get_session_log_path(event.session_id)
            event_dict: Dict[str, Any] = {}
            for k, v in asdict(event).items():
                if v is not None:
                    event_dict[k] = v

            # Standardize event field
            if "event" not in event_dict:
                event_dict["event"] = getattr(event, "event_type", "UNKNOWN")
            event_dict["event_type"] = event_dict["event"]

            line = json.dumps(event_dict)
            with self._lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass

    def log_t0_gate(
        self,
        session_id: str,
        delta_similarity: float,
        is_stable: bool,
        latency_ms: float,
        decision_action: Optional[str] = None,
        answer_version: Optional[int] = None
    ) -> None:
        """Helper to log T0 intent stability evaluation."""
        action = decision_action or ("PROCEED" if is_stable else "WAIT")
        event = TelemetryEvent(
            session_id=session_id,
            event="T0_GATE",
            latency_ms=latency_ms,
            delta_similarity=delta_similarity,
            is_stable=is_stable,
            decision_action=action,
            answer_version=answer_version
        )
        self.log_event(event)

    def log_t1_decision(
        self,
        session_id: str,
        retrieve: bool,
        is_refine: bool,
        sub_queries: List[str],
        answer_version: int,
        latency_ms: float,
        reasoning: Optional[str] = None,
        decision_action: Optional[str] = None
    ) -> None:
        """Helper to log T1 LLM routing and multi-intent decomposition decision."""
        action = decision_action or ("REFINE" if is_refine else ("RETRIEVE" if retrieve else "NO_RETRIEVAL"))
        event = TelemetryEvent(
            session_id=session_id,
            event="T1_DECISION",
            latency_ms=latency_ms,
            should_retrieve=retrieve,
            retrieval_decision=retrieve,
            is_refine=is_refine,
            sub_queries=sub_queries,
            answer_version=answer_version,
            reasoning=reasoning,
            decision_action=action
        )
        self.log_event(event)

    def log_retrieval(
        self,
        session_id: str,
        latency_ms: float,
        number_of_subqueries: int,
        evidence_count: int,
        retrieval_call_count: int,
        answer_version: Optional[int] = None
    ) -> None:
        """Helper to log overall multi-intent retrieval metrics."""
        event = TelemetryEvent(
            session_id=session_id,
            event="RETRIEVAL",
            latency_ms=latency_ms,
            number_of_subqueries=number_of_subqueries,
            evidence_count=evidence_count,
            retrieval_call_count=retrieval_call_count,
            answer_version=answer_version
        )
        self.log_event(event)

    def log_retrieval_stage(
        self,
        session_id: str,
        stage_event: str,
        latency_ms: float,
        sub_query_id: Optional[int] = None,
        query: Optional[str] = None,
        result_count: Optional[int] = None,
        answer_version: Optional[int] = None
    ) -> None:
        """Helper to log stage-level retrieval timings (RETRIEVAL_BM25, RETRIEVAL_DENSE, RETRIEVAL_RRF)."""
        event = TelemetryEvent(
            session_id=session_id,
            event=stage_event,
            latency_ms=latency_ms,
            sub_query_id=sub_query_id,
            query=query,
            result_count=result_count,
            answer_version=answer_version
        )
        self.log_event(event)

    def log_retrieval_chunk(
        self,
        session_id: str,
        sub_query_id: int,
        chunk_id: str,
        method: str,
        rank: int,
        score: float,
        reranker_score: Optional[float] = None,
        doc_id: Optional[str] = None,
        section: Optional[str] = None,
        answer_version: Optional[int] = None
    ) -> None:
        """Helper to log per-chunk retrieval, fusion, and reranking stats."""
        event = TelemetryEvent(
            session_id=session_id,
            event="RETRIEVAL_CHUNK",
            sub_query_id=sub_query_id,
            chunk_id=chunk_id,
            retrieval_method=method,
            rank=rank,
            score=score,
            reranker_score=reranker_score,
            doc_id=doc_id,
            section=section,
            answer_version=answer_version
        )
        self.log_event(event)

    def log_synthesis(
        self,
        session_id: str,
        latency_ms: float,
        claims_generated: int,
        citations_generated: int,
        answer_version: Optional[int] = None,
        is_refinement: Optional[bool] = None,
        usage_available: bool = False,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None
    ) -> None:
        """Helper to log structured claim synthesis metrics."""
        total_tokens = (tokens_prompt + tokens_completion) if (tokens_prompt is not None and tokens_completion is not None) else None
        event = TelemetryEvent(
            session_id=session_id,
            event="SYNTHESIS",
            latency_ms=latency_ms,
            claims_generated=claims_generated,
            citations_generated=citations_generated,
            answer_version=answer_version,
            is_refinement=is_refinement,
            usage_available=usage_available,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            total_tokens=total_tokens
        )
        self.log_event(event)

    def log_claim_verification(
        self,
        session_id: str,
        claim_id: int,
        claim_text: str,
        result: str,
        citations: List[str],
        method: str,
        latency_ms: float,
        citation_count: Optional[int] = None,
        answer_version: Optional[int] = None
    ) -> None:
        """Helper to log claim verification and grounding results."""
        c_count = citation_count if citation_count is not None else len(citations)
        event = TelemetryEvent(
            session_id=session_id,
            event="CLAIM_VERIFY",
            claim_id=claim_id,
            claim_text=claim_text,
            claim_verification_result=result,
            citations=citations,
            citation_count=c_count,
            verification_method=method,
            latency_ms=latency_ms,
            answer_version=answer_version
        )
        self.log_event(event)

    def log_e2e(
        self,
        session_id: str,
        total_latency_ms: float,
        controller_latency_ms: float,
        retrieval_latency_ms: float,
        synthesis_latency_ms: float,
        verification_latency_ms: float,
        answer_version: Optional[int] = None
    ) -> None:
        """Helper to log complete end-to-end turn latencies."""
        event = TelemetryEvent(
            session_id=session_id,
            event="E2E",
            total_latency_ms=total_latency_ms,
            controller_latency_ms=controller_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            synthesis_latency_ms=synthesis_latency_ms,
            verification_latency_ms=verification_latency_ms,
            latency_ms=total_latency_ms,
            answer_version=answer_version
        )
        self.log_event(event)


# Global telemetry logger instance placeholder
telemetry_logger = TelemetryLogger()
