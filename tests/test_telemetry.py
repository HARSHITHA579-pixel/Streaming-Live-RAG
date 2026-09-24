"""
Structured Telemetry and Observability Test Suite (Milestone 9)

================================================================================
TEST COVERAGE:
1. JSONL file creation and formatting
2. Independent JSON validity for each logged event line
3. T0 Gate telemetry event recording (T0_GATE)
4. T1 Decision telemetry event recording (T1_DECISION)
5. Retrieval telemetry event recording (RETRIEVAL)
6. Sub-stage retrieval telemetry events (RETRIEVAL_BM25, RETRIEVAL_DENSE, RETRIEVAL_RRF)
7. Synthesis telemetry event recording (SYNTHESIS)
8. Claim verification telemetry event recording (CLAIM_VERIFY)
9. Answer version progression across multiple turns (v1 -> v2)
10. End-to-end latency event recording (E2E)
11. Multi-session isolation (logs/session_<id>.jsonl separation)
12. HTTP Telemetry endpoint validation (GET /telemetry/{session_id})
================================================================================
"""

import os
import sys
import json
import pytest
from typing import cast, Any
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from app.main import app, orchestrator, active_sessions, session_transcript_buffers
from app.telemetry import TelemetryLogger, TelemetryEvent, telemetry_logger
from app.config import settings


@pytest.fixture
def temp_telemetry_logger(tmp_path):
    """Fixture providing an isolated TelemetryLogger writing to a temporary directory."""
    return TelemetryLogger(log_dir=str(tmp_path / "logs"))


def test_jsonl_file_creation_and_valid_json(temp_telemetry_logger):
    """
    1 & 2. Verify JSONL file creation and that every line is valid, independently parseable JSON.
    """
    session_id = "test_jsonl_creation"
    
    event1 = TelemetryEvent(
        session_id=session_id,
        event="T0_GATE",
        latency_ms=1.5,
        is_stable=True,
        delta_similarity=0.92
    )
    event2 = TelemetryEvent(
        session_id=session_id,
        event="T1_DECISION",
        latency_ms=3.2,
        should_retrieve=True,
        is_refine=False,
        sub_queries=["What is the hotel reimbursement policy?"],
        answer_version=1
    )
    
    temp_telemetry_logger.log_event(event1)
    temp_telemetry_logger.log_event(event2)
    
    log_path = temp_telemetry_logger.get_session_log_path(session_id)
    assert log_path.exists(), "Telemetry log file was not created"
    
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        assert "timestamp" in parsed
        assert "session_id" in parsed
        assert parsed["session_id"] == session_id
        assert "event" in parsed


def test_t0_telemetry_event(temp_telemetry_logger):
    """
    3. Verify T0_GATE telemetry captures is_stable, delta_similarity, latency_ms, decision_action.
    """
    session_id = "test_t0_event"
    temp_telemetry_logger.log_t0_gate(
        session_id=session_id,
        delta_similarity=0.35,
        is_stable=False,
        latency_ms=2.1,
        decision_action="WAIT",
        answer_version=0
    )
    
    log_path = temp_telemetry_logger.get_session_log_path(session_id)
    with open(log_path, "r", encoding="utf-8") as f:
        event = json.loads(f.readline().strip())
        
    assert event["event"] == "T0_GATE"
    assert event["session_id"] == session_id
    assert event["is_stable"] is False
    assert event["delta_similarity"] == 0.35
    assert event["latency_ms"] == 2.1
    assert event["decision_action"] == "WAIT"


def test_t1_telemetry_event(temp_telemetry_logger):
    """
    4. Verify T1_DECISION telemetry captures should_retrieve, is_refine, sub_queries, answer_version.
    """
    session_id = "test_t1_event"
    temp_telemetry_logger.log_t1_decision(
        session_id=session_id,
        retrieve=True,
        is_refine=False,
        sub_queries=["Query A", "Query B"],
        answer_version=1,
        latency_ms=4.5,
        reasoning="Multi-intent query detected"
    )
    
    log_path = temp_telemetry_logger.get_session_log_path(session_id)
    with open(log_path, "r", encoding="utf-8") as f:
        event = json.loads(f.readline().strip())
        
    assert event["event"] == "T1_DECISION"
    assert event["should_retrieve"] is True
    assert event["is_refine"] is False
    assert event["sub_queries"] == ["Query A", "Query B"]
    assert event["answer_version"] == 1
    assert event["latency_ms"] == 4.5
    assert event["decision_action"] == "RETRIEVE"


def test_retrieval_and_stage_events(temp_telemetry_logger):
    """
    5 & 6. Verify RETRIEVAL, RETRIEVAL_BM25, RETRIEVAL_DENSE, and RETRIEVAL_RRF telemetry.
    """
    session_id = "test_retrieval_stages"
    
    # Sub-stage events
    temp_telemetry_logger.log_retrieval_stage(
        session_id=session_id,
        stage_event="RETRIEVAL_BM25",
        latency_ms=1.4,
        sub_query_id=0,
        query="hotel rate",
        result_count=4
    )
    temp_telemetry_logger.log_retrieval_stage(
        session_id=session_id,
        stage_event="RETRIEVAL_DENSE",
        latency_ms=25.8,
        sub_query_id=0,
        query="hotel rate",
        result_count=4
    )
    temp_telemetry_logger.log_retrieval_stage(
        session_id=session_id,
        stage_event="RETRIEVAL_RRF",
        latency_ms=0.6,
        sub_query_id=0,
        result_count=4
    )
    
    # Overall retrieval event
    temp_telemetry_logger.log_retrieval(
        session_id=session_id,
        latency_ms=28.5,
        number_of_subqueries=1,
        evidence_count=4,
        retrieval_call_count=1,
        answer_version=1
    )
    
    # Chunk event
    temp_telemetry_logger.log_retrieval_chunk(
        session_id=session_id,
        sub_query_id=0,
        chunk_id="hotel_policy_01",
        method="rrf",
        rank=1,
        score=0.0325,
        doc_id="hotel_policy",
        section="Nightly Caps"
    )
    
    log_path = temp_telemetry_logger.get_session_log_path(session_id)
    with open(log_path, "r", encoding="utf-8") as f:
        events = [json.loads(line.strip()) for line in f if line.strip()]
        
    event_types = [e["event"] for e in events]
    assert "RETRIEVAL_BM25" in event_types
    assert "RETRIEVAL_DENSE" in event_types
    assert "RETRIEVAL_RRF" in event_types
    assert "RETRIEVAL" in event_types
    assert "RETRIEVAL_CHUNK" in event_types
    
    ret_ev = next(e for e in events if e["event"] == "RETRIEVAL")
    assert ret_ev["evidence_count"] == 4
    assert ret_ev["number_of_subqueries"] == 1


def test_synthesis_and_claim_verify_telemetry(temp_telemetry_logger):
    """
    7 & 8. Verify SYNTHESIS and CLAIM_VERIFY telemetry recording.
    """
    session_id = "test_synth_verify"
    
    temp_telemetry_logger.log_synthesis(
        session_id=session_id,
        latency_ms=6.2,
        claims_generated=3,
        citations_generated=3,
        answer_version=1,
        is_refinement=False,
        usage_available=False
    )
    
    temp_telemetry_logger.log_claim_verification(
        session_id=session_id,
        claim_id=1,
        claim_text="Hotel rate is capped at $220/night.",
        result="SUPPORTED",
        citations=["[DOC: hotel_policy | Section: Caps | Chunk: hotel_policy_01]"],
        method="cheap_local",
        latency_ms=0.8,
        answer_version=1
    )
    
    log_path = temp_telemetry_logger.get_session_log_path(session_id)
    with open(log_path, "r", encoding="utf-8") as f:
        events = [json.loads(line.strip()) for line in f if line.strip()]
        
    synth_ev = next(e for e in events if e["event"] == "SYNTHESIS")
    assert synth_ev["claims_generated"] == 3
    assert synth_ev["citations_generated"] == 3
    assert synth_ev["usage_available"] is False
    
    claim_ev = next(e for e in events if e["event"] == "CLAIM_VERIFY")
    assert claim_ev["claim_id"] == 1
    assert claim_ev["claim_verification_result"] == "SUPPORTED"
    assert claim_ev["verification_method"] == "cheap_local"
    assert claim_ev["citation_count"] == 1


def test_e2e_latency_telemetry(temp_telemetry_logger):
    """
    10. Verify E2E telemetry event capturing total and component latencies.
    """
    session_id = "test_e2e_event"
    temp_telemetry_logger.log_e2e(
        session_id=session_id,
        total_latency_ms=55.4,
        controller_latency_ms=2.1,
        retrieval_latency_ms=42.0,
        synthesis_latency_ms=6.5,
        verification_latency_ms=4.8,
        answer_version=1
    )
    
    log_path = temp_telemetry_logger.get_session_log_path(session_id)
    with open(log_path, "r", encoding="utf-8") as f:
        event = json.loads(f.readline().strip())
        
    assert event["event"] == "E2E"
    assert event["total_latency_ms"] == 55.4
    assert event["controller_latency_ms"] == 2.1
    assert event["retrieval_latency_ms"] == 42.0
    assert event["synthesis_latency_ms"] == 6.5
    assert event["verification_latency_ms"] == 4.8
    assert event["answer_version"] == 1


@pytest.mark.asyncio
async def test_answer_version_progression_in_telemetry():
    """
    9. Verify answer_version progresses from v1 -> v2 in recorded pipeline telemetry across turns.
    """
    active_sessions.clear()
    session_transcript_buffers.clear()
    session_id = "test_telemetry_versioning"
    
    # Turn 1
    await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="What is the corporate travel reimbursement submission window?"
    )
    
    # Turn 2 (Refinement)
    await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="Actually, this is for an international business trip."
    )
    
    log_path = telemetry_logger.get_session_log_path(session_id)
    assert log_path.exists()
    
    with open(log_path, "r", encoding="utf-8") as f:
        events = [json.loads(line.strip()) for line in f if line.strip()]
        
    versions = [e["answer_version"] for e in events if "answer_version" in e and e["answer_version"] is not None]
    assert 1 in versions
    assert 2 in versions
    assert versions[-1] == 2


@pytest.mark.asyncio
async def test_session_isolation_in_telemetry():
    """
    11. Verify session isolation: session_A and session_B produce distinct log files with no bleed.
    """
    active_sessions.clear()
    session_transcript_buffers.clear()
    
    session_a = "telemetry_user_alpha"
    session_b = "telemetry_user_beta"
    
    await orchestrator.process_streaming_transcript(
        session_id=session_a,
        transcript_chunk="What is the hotel reimbursement rate?"
    )
    await orchestrator.process_streaming_transcript(
        session_id=session_b,
        transcript_chunk="What is the cancellation policy for events?"
    )
    
    path_a = telemetry_logger.get_session_log_path(session_a)
    path_b = telemetry_logger.get_session_log_path(session_b)
    
    assert path_a.exists()
    assert path_b.exists()
    assert path_a != path_b
    
    with open(path_a, "r", encoding="utf-8") as f:
        events_a = [json.loads(line.strip()) for line in f if line.strip()]
    with open(path_b, "r", encoding="utf-8") as f:
        events_b = [json.loads(line.strip()) for line in f if line.strip()]
        
    assert all(e["session_id"] == session_a for e in events_a)
    assert all(e["session_id"] == session_b for e in events_b)


def test_telemetry_http_endpoint():
    """
    12. Verify GET /telemetry/{session_id} returns all recorded JSONL events as a structured JSON response.
    """
    client = TestClient(cast(Any, app))
    session_id = "test_endpoint_session"
    
    # Send a query via WebSocket to produce telemetry
    with client.websocket_connect(f"/ws/stream/{session_id}") as ws:
        ws.send_text("What are the hotel reimbursement limits?")
        while True:
            evt = ws.receive_json()
            if evt.get("event") == "answer":
                break
                
    response = client.get(f"/telemetry/{session_id}")
    assert response.status_code == 200
    data = response.json()
    
    assert data["session_id"] == session_id
    assert data["count"] > 0
    assert len(data["events"]) == data["count"]
    
    event_names = [e.get("event") for e in data["events"]]
    assert "T0_GATE" in event_names
    assert "T1_DECISION" in event_names
    assert "RETRIEVAL" in event_names
    assert "SYNTHESIS" in event_names
    assert "E2E" in event_names
