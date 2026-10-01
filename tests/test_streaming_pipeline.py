"""
Validates the End-to-End Streaming Pipeline and WebSocket Integration in Streaming Live RAG.
Ensures live transcript streams, multi-turn conversational refinements, delta retrievals,
and answer versioning execute reliably across concurrent isolated sessions without state leakage.
Demonstrates: Full streaming lifecycle, WebSocket protocol frames, session state continuity, and versioned answer updates.
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
async def test_test1_additive_multi_intent():
    """
    TEST 1 — Additive multi-intent
    Input: "hotel reimbursement and cancellation policy"
    Expected: Both intents remain active and final answer can contain both.
    """
    session_id = "test1_additive"
    query = "hotel reimbursement and cancellation policy"

    events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk=query
    )

    ctrl_event = next(e for e in events if e["event"] == "controller")
    assert ctrl_event["t0_stable"] is True
    assert ctrl_event["should_retrieve"] is True
    assert len(ctrl_event["sub_queries"]) >= 2

    session = active_sessions[session_id]
    assert len(session.active_intents) >= 2
    assert any("hotel" in i.lower() for i in session.active_intents)
    assert any("cancellation" in i.lower() for i in session.active_intents)

    ans_event = next(e for e in events if e["event"] == "answer")
    assert ans_event["answer_version"] == 1


@pytest.mark.asyncio
async def test_test2_cross_turn_refinement():
    """
    TEST 2 — Cross-turn refinement
    Turn 1: "Tell me about Pune travel."
    Turn 2: "Actually, I mean international travel."
    Expected:
    - Same session_id
    - answer_version increases
    - Pune evidence becomes stale/superseded
    - International evidence active
    - Final answer does NOT contain Pune-specific policy information
    """
    session_id = "test2_cross_refine"

    # Turn 1
    t1_events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="Tell me about Pune travel."
    )
    ans_t1 = next(e for e in t1_events if e["event"] == "answer")
    assert ans_t1["answer_version"] == 1
    session = active_sessions[session_id]

    # Turn 2
    t2_events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="Actually, I mean international travel."
    )
    ans_t2 = next(e for e in t2_events if e["event"] == "answer")
    assert ans_t2["answer_version"] == 2
    assert session.session_id == session_id

    # Active evidence should only contain international travel
    assert len(session.active_evidence_pool) > 0
    for chunk in session.active_evidence_pool.values():
        assert "pune" not in chunk.text.lower()

    # Final answer should not contain Pune policy info
    assert "pune" not in ans_t2["answer"].lower()
    assert "international" in ans_t2["answer"].lower()


@pytest.mark.asyncio
async def test_test3_same_turn_refinement():
    """
    TEST 3 — Same-turn refinement
    Input: "Pune travel, actually international travel."
    Expected:
    - Pune is superseded.
    - International is active.
    """
    session_id = "test3_same_turn"
    events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="Pune travel, actually international travel."
    )

    ctrl_event = next(e for e in events if e["event"] == "controller")
    assert ctrl_event["is_refine"] is True
    assert ctrl_event["is_supersession"] is True

    session = active_sessions[session_id]
    assert any("international" in i.lower() for i in session.active_intents)
    assert not any("pune" in i.lower() for i in session.active_intents)

    ans_event = next(e for e in events if e["event"] == "answer")
    assert "international" in ans_event["answer"].lower()
    assert "pune" not in ans_event["answer"].lower()


@pytest.mark.asyncio
async def test_test4_continuation():
    """
    TEST 4 — Continuation
    Turn 1: "Tell me about hotel reimbursement."
    Turn 2: "for 30 people"
    Expected:
    - Hotel reimbursement remains active and gets enriched.
    - It is NOT replaced.
    """
    session_id = "test4_continuation"

    # Turn 1
    await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="Tell me about hotel reimbursement."
    )
    session = active_sessions[session_id]
    assert any("hotel" in i.lower() for i in session.active_intents)

    # Turn 2
    t2_events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="for 30 people"
    )
    ctrl_t2 = next(e for e in t2_events if e["event"] == "controller")
    assert ctrl_t2["is_refine"] is True
    assert ctrl_t2["is_continuation"] is True
    assert session.answer_version == 2

    # Intent is enriched and preserved
    assert any("hotel" in i.lower() or "30 people" in i.lower() for i in session.active_intents)
    assert len(session.superseded_evidence_pool) == 0  # Not superseded


@pytest.mark.asyncio
async def test_test5_explicit_removal():
    """
    TEST 5 — Explicit removal
    Turn 1: "hotel reimbursement and cancellation policy"
    Turn 2: "Forget hotel reimbursement, only cancellation."
    Expected:
    - Only cancellation remains active.
    - Hotel reimbursement removed from active context.
    """
    session_id = "test5_removal"

    # Turn 1
    await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="hotel reimbursement and cancellation policy"
    )
    session = active_sessions[session_id]
    assert any("hotel" in i.lower() for i in session.active_intents)
    assert any("cancellation" in i.lower() for i in session.active_intents)

    # Turn 2
    t2_events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="Forget hotel reimbursement, only cancellation."
    )
    ctrl_t2 = next(e for e in t2_events if e["event"] == "controller")
    assert ctrl_t2["is_refine"] is True
    assert ctrl_t2["is_removal"] is True

    # Check active intents and active evidence
    assert any("cancellation" in i.lower() for i in session.active_intents)
    assert not any("hotel" in i.lower() for i in session.active_intents)

    for chunk in session.active_evidence_pool.values():
        if "cancellation" in chunk.text.lower():
            continue
        assert "hotel nightly rate" not in chunk.text.lower()


@pytest.mark.asyncio
async def test_test6_no_stale_evidence_leakage():
    """
    TEST 6 — No stale evidence leakage
    Old: "domestic travel"
    New: "Actually, I mean international travel."
    Assert that old domestic-only facts ($85 domestic per diem, 14 days domestic booking,
    domestic rail, $220 domestic hotel cap) do NOT appear in the refined answer.
    Assert that international facts (30 days, $125 or $95 per diem, business class) DO appear.
    """
    session_id = "test6_leakage"

    # Turn 1: Domestic travel query
    t1_events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="What are the domestic travel guidelines, domestic per diem, and booking timelines?"
    )
    ans_t1 = next(e for e in t1_events if e["event"] == "answer")
    assert ans_t1["answer_version"] == 1
    assert "travel" in ans_t1["answer"].lower() or "$85" in ans_t1["answer"] or "domestic" in ans_t1["answer"].lower()

    # Turn 2: Superseded by international travel
    t2_events = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="Actually, I mean international travel."
    )
    ans_t2 = next(e for e in t2_events if e["event"] == "answer")
    assert ans_t2["answer_version"] == 2
    refined_answer = ans_t2["answer"]

    # Strict assertion: NO domestic-only policy facts in the refined v2 answer
    domestic_stale_facts = [
        "$85",
        "14 days",
        "14 calendar days prior to domestic",
        "domestic rail",
        "$220",
        "$0.67 per mile"
    ]
    for stale_fact in domestic_stale_facts:
        assert stale_fact not in refined_answer, f"Stale domestic fact '{stale_fact}' leaked into refined answer:\n{refined_answer}"

    # International facts MUST be present
    assert any(term in refined_answer.lower() for term in ["international", "30 calendar days", "30 days", "$125", "$95", "business class", "passport", "visa"])


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


@pytest.mark.asyncio
async def test_same_turn_multisegment_retrieval_and_synthesis():
    """
    11. Verify same-turn multi-segment voice streaming:
    Segment 1 (is_final=False) -> background retrieval -> evidence accumulated
    Segment 2 (is_final=False) -> background retrieval -> evidence pool expanded
    Final Stop (is_final=True) -> unified synthesis covering all supported segments
    """
    session_id = "test_multiseg_turn"

    # Segment 1: Fire safety temporary structures
    ev1 = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="What are the fire safety requirements for temporary event structures?",
        is_final=False
    )
    # No premature answer in segment 1
    assert not any(e["event"] == "answer" for e in ev1)
    session = active_sessions[session_id]
    pool_size_1 = len(session.evidence_pool)
    assert pool_size_1 > 0

    # Segment 2: Crowd management mass gathering
    ev2 = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk="What crowd control measures are required for a mass gathering?",
        is_final=False
    )
    # No premature answer in segment 2
    assert not any(e["event"] == "answer" for e in ev2)
    pool_size_2 = len(session.evidence_pool)
    assert pool_size_2 >= pool_size_1

    # Final Stop: Complete transcript with both topics
    full_turn = "What are the fire safety requirements for temporary event structures and what crowd control measures are required for a mass gathering?"
    ev_final = await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk=full_turn,
        is_final=True
    )
    ans_evt = next((e for e in ev_final if e["event"] == "answer"), None)
    assert ans_evt is not None
    assert ans_evt["answer_version"] == 1
    assert len(ans_evt["citations"]) >= 2
    # Check that both fire and crowd/event topics are cited
    assert any("fire" in c.lower() or "safety" in c.lower() or "event" in c.lower() or "guide" in c.lower() for c in ans_evt["citations"])
    assert any("crowd" in c.lower() or "gathering" in c.lower() or "event" in c.lower() or "guide" in c.lower() for c in ans_evt["citations"])


