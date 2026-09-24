"""
End-to-End Streaming Pipeline and WebSocket Integration Tests

================================================================================
TEST COVERAGE:
1. Partial transcript -> WAIT (T0 Gate prevents premature retrieval)
2. Stable single-intent query -> RETRIEVAL + GROUNDED ANSWER
3. Multi-intent query -> Multi-subquery decomposition + balanced retrieval
4. Refinement turn -> Delta retrieval + Session state preservation + answer_version increment
5. Conversational turn -> No-retrieval bypass
6. Unsupported query -> Factual grounding refusal / uncertainty note
7. Citation preservation across pipeline and turns
8. Session isolation between concurrent sessions
9. WebSocket streaming endpoint via TestClient (asyncio.Queue ingestion & frame emission)
10. Telemetry generation and HTTP inspection endpoints
================================================================================
"""

import sys
import json
import asyncio
from pathlib import Path

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
from typing import cast, Any
from fastapi.testclient import TestClient

from app.main import app, orchestrator, active_sessions, session_transcript_buffers
from app.session_state import SessionState, ClaimStatus
from app.config import settings


@pytest.fixture(autouse=True)
def clean_sessions():
    """Clear in-memory session states before each test."""
    active_sessions.clear()
    session_transcript_buffers.clear()
    yield
    active_sessions.clear()
    session_transcript_buffers.clear()


@pytest.mark.asyncio
async def test_partial_transcript_wait():
    """
    1. Partial transcript -> T0 unstable -> WAIT
    No retrieval, no claims, no final answer event.
    """
    session_id = "test_wait_session"
    events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="What is the..."
    )

    event_types = [e["event"] for e in events]
    assert "transcript" in event_types
    assert "controller" in event_types
    assert "retrieval" not in event_types
    assert "claim" not in event_types
    assert "answer" not in event_types

    ctrl_event = next(e for e in events if e["event"] == "controller")
    assert ctrl_event["t0_stable"] is False
    assert ctrl_event["should_retrieve"] is False


@pytest.mark.asyncio
async def test_stable_single_intent_retrieval():
    """
    2. Stable single-intent query -> RETRIEVAL -> CLAIMS -> VERIFICATION -> ANSWER
    """
    session_id = "test_single_intent"
    query = "What is the corporate hotel nightly reimbursement rate limit?"

    events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk=query
    )

    event_types = [e["event"] for e in events]
    assert "transcript" in event_types
    assert "controller" in event_types
    assert "retrieval" in event_types
    assert "claim" in event_types
    assert "verification" in event_types
    assert "answer" in event_types

    ctrl_event = next(e for e in events if e["event"] == "controller")
    assert ctrl_event["t0_stable"] is True
    assert ctrl_event["should_retrieve"] is True
    assert len(ctrl_event["sub_queries"]) == 1

    ret_event = next(e for e in events if e["event"] == "retrieval")
    assert ret_event["evidence_count"] > 0
    assert len(ret_event["chunks"]) > 0

    ans_event = next(e for e in events if e["event"] == "answer")
    assert ans_event["answer_version"] == 1
    assert len(ans_event["answer"]) > 0
    assert len(ans_event["citations"]) > 0

    # Verify session state persisted
    session = active_sessions[session_id]
    assert session.answer_version == 1
    assert len(session.evidence_pool) > 0
    assert len(session.claims) > 0


@pytest.mark.asyncio
async def test_multi_intent_decomposition():
    """
    3. Multi-intent query -> multiple sub-queries and evidence for both intents
    """
    session_id = "test_multi_intent"
    query = "What are the hotel nightly rate limits and what is the cancellation policy for events?"

    events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk=query
    )

    ctrl_event = next(e for e in events if e["event"] == "controller")
    assert ctrl_event["t0_stable"] is True
    assert ctrl_event["should_retrieve"] is True
    assert len(ctrl_event["sub_queries"]) >= 2

    ret_event = next(e for e in events if e["event"] == "retrieval")
    assert ret_event["evidence_count"] >= 2

    ans_event = next(e for e in events if e["event"] == "answer")
    assert ans_event["answer_version"] == 1


@pytest.mark.asyncio
async def test_refinement_delta_retrieval():
    """
    4. Refinement -> delta retrieval, state preserved, answer_version incremented
    """
    session_id = "test_refinement"

    # Turn 1: Initial query
    turn1_query = "What is the expense reimbursement submission timeline?"
    events_t1 = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk=turn1_query
    )
    ans_t1 = next(e for e in events_t1 if e["event"] == "answer")
    assert ans_t1["answer_version"] == 1
    evidence_count_t1 = len(active_sessions[session_id].evidence_pool)

    # Turn 2: Follow-up refinement
    turn2_query = "Actually, I mean international travel currency conversion rules."
    events_t2 = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk=turn2_query
    )

    ctrl_t2 = next(e for e in events_t2 if e["event"] == "controller")
    assert ctrl_t2["is_refine"] is True
    assert ctrl_t2["should_retrieve"] is True

    ans_t2 = next(e for e in events_t2 if e["event"] == "answer")
    assert ans_t2["answer_version"] == 2
    assert ans_t2["is_refinement"] is True

    # Evidence pool should accumulate (not wipe out)
    evidence_count_t2 = len(active_sessions[session_id].evidence_pool)
    assert evidence_count_t2 >= evidence_count_t1


@pytest.mark.asyncio
async def test_conversational_no_retrieval():
    """
    5. Conversational turns -> retrieve=False -> No corpus search
    """
    session_id = "test_conversational"

    conversational_inputs = ["Hello!", "Thank you, that answers my question."]
    for text in conversational_inputs:
        events = await orchestrator.process_streaming_transcript(
            session_id=session_id,
            transcript_chunk=text
        )
        ctrl_event = next(e for e in events if e["event"] == "controller")
        assert ctrl_event["should_retrieve"] is False

        # Retrieval event must NOT be emitted
        assert not any(e["event"] == "retrieval" for e in events)

        # Answer must be emitted
        ans_event = next(e for e in events if e["event"] == "answer")
        assert len(ans_event["answer"]) > 0


@pytest.mark.asyncio
async def test_unsupported_query_uncertainty():
    """
    6. Unsupported query -> Diverted to uncertainty, no hallucinated verified fact
    """
    session_id = "test_unsupported"
    query = "What is the cryptocurrency reimbursement and pet travel policy?"

    events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk=query
    )

    ans_event = next(e for e in events if e["event"] == "answer")
    # Verify uncertainty section or note is present
    session = active_sessions[session_id]
    assert len(session.uncertainty_notes) > 0 or "Uncertainty" in ans_event["answer"] or "No verified" in ans_event["answer"]


@pytest.mark.asyncio
async def test_citation_preservation():
    """
    7. Citation format [DOC: ... | Section: ... | Chunk: ...] preservation
    """
    session_id = "test_citations"
    query = "What are the rules for travel reimbursement and hotel bookings?"

    events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk=query
    )

    ans_event = next(e for e in events if e["event"] == "answer")
    citations = ans_event["citations"]
    assert len(citations) > 0
    for cite in citations:
        assert "[DOC:" in cite or "Chunk:" in cite


@pytest.mark.asyncio
async def test_session_isolation():
    """
    8. Multi-session isolation: Session A and Session B maintain independent states
    """
    session_a = "session_alice"
    session_b = "session_bob"

    await orchestrator.process_streaming_transcript(
        session_id=session_a,
        transcript_chunk="What is the hotel reimbursement policy?"
    )

    await orchestrator.process_streaming_transcript(
        session_id=session_b,
        transcript_chunk="What is the event cancellation penalty?"
    )

    state_a = active_sessions[session_a]
    state_b = active_sessions[session_b]

    assert state_a.session_id == session_a
    assert state_b.session_id == session_b
    assert state_a is not state_b

    # Verify covered intents are separate
    assert any("hotel" in intent.lower() for intent in state_a.covered_intents)
    assert any("cancellation" in intent.lower() or "event" in intent.lower() for intent in state_b.covered_intents)


def test_http_and_websocket_endpoints():
    """
    9. Test HTTP health, session inspection, and WebSocket streaming endpoints using TestClient
    """
    client = TestClient(cast(Any, app))

    # Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

    # WebSocket connection & streaming interaction
    session_id = "ws_test_session"
    with client.websocket_connect(f"/ws/stream/{session_id}") as ws:
        # Send partial chunk
        ws.send_text("What is the...")
        # Receive events for partial chunk
        msg1 = ws.receive_json()
        assert msg1["event"] == "transcript"
        msg2 = ws.receive_json()
        assert msg2["event"] == "controller"
        assert msg2["t0_stable"] is False

        # Send complete query
        ws.send_text("What is the corporate travel reimbursement policy?")
        received_events = []
        # Receive until answer event is received
        while True:
            evt = ws.receive_json()
            received_events.append(evt)
            if evt.get("event") == "answer":
                break

        event_names = [e["event"] for e in received_events]
        assert "transcript" in event_names
        assert "controller" in event_names
        assert "retrieval" in event_names
        assert "claim" in event_names
        assert "verification" in event_names
        assert "answer" in event_names

    # Inspect session via HTTP
    resp_session = client.get(f"/session/{session_id}")
    assert resp_session.status_code == 200
    s_data = resp_session.json()
    assert s_data["session_id"] == session_id
    assert s_data["answer_version"] >= 1

    # Reset session via HTTP
    resp_reset = client.post(f"/session/{session_id}/reset")
    assert resp_reset.status_code == 200
    assert resp_reset.json()["status"] == "reset_successful"

    resp_session_after = client.get(f"/session/{session_id}")
    assert resp_session_after.json()["answer_version"] == 0


def test_telemetry_endpoint():
    """
    10. Verify GET /telemetry/{session_id} returns logged JSONL events
    """
    client = TestClient(cast(Any, app))
    session_id = "telemetry_test_session"

    # Run a query to generate telemetry
    with client.websocket_connect(f"/ws/stream/{session_id}") as ws:
        ws.send_text("What is the hotel reimbursement policy?")
        while True:
            evt = ws.receive_json()
            if evt.get("event") == "answer":
                break

    # Fetch telemetry
    resp = client.get(f"/telemetry/{session_id}")
    assert resp.status_code == 200
    telemetry_data = resp.json()
    assert telemetry_data["session_id"] == session_id
    assert telemetry_data["count"] > 0
    assert len(telemetry_data["events"]) > 0
