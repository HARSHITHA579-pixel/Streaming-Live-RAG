import os
import sys
import asyncio
from pathlib import Path

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
from app.controller import TwoTierController, ControllerDecision
from app.session_state import SessionState, StructuredClaim, ClaimStatus
from app.config import settings


async def run_controller_tests():
    print("\n=======================================================")
    print("TWO-TIER CONTROLLER TEST SUITE")
    print("=======================================================\n")

    controller = TwoTierController(app_settings=settings)

    # -----------------------------------------------------------------
    # TEST A: Incomplete Utterance -> WAIT (T0 Gate)
    # -----------------------------------------------------------------
    print("-------------------------------------------------------")
    print("TEST A: Incomplete Utterance -> WAIT (T0 Gate)")
    print("-------------------------------------------------------")
    incomplete_chunks = [
        ("What", ""),
        ("What is the...", "What"),
        ("Can I book hotel for", "Can I book hotel"),
    ]
    empty_state = SessionState(session_id="test_session_a")

    for curr, prev in incomplete_chunks:
        decision = await controller.process_incoming_transcript(curr, prev, empty_state)
        print(f"Transcript: \"{curr}\" (prev: \"{prev}\")")
        print(f"  -> T0 Stable: {decision.t0_stable} | Retrieve: {decision.should_retrieve}")
        print(f"  -> Decision: {decision.reasoning}")
        assert decision.t0_stable is False, f"Expected T0 unstable for '{curr}'"
        assert decision.should_retrieve is False
    print("TEST A PASSED [WAIT on partial speech]\n")

    # -----------------------------------------------------------------
    # TEST B: Stable Single-Intent Query -> RETRIEVE (1 Subquery)
    # -----------------------------------------------------------------
    print("-------------------------------------------------------")
    print("TEST B: Stable Single-Intent Query -> RETRIEVE (1 Subquery)")
    print("-------------------------------------------------------")
    single_intent_query = "What is the corporate travel reimbursement policy?"
    decision_b = await controller.process_incoming_transcript(
        current_transcript=single_intent_query,
        previous_transcript="",
        session_state=empty_state
    )
    print(f"Query: \"{single_intent_query}\"")
    print(f"  -> T0 Stable: {decision_b.t0_stable}")
    print(f"  -> Retrieve: {decision_b.should_retrieve}")
    print(f"  -> Is Refine: {decision_b.is_refine}")
    print(f"  -> Sub-Queries ({len(decision_b.sub_queries)}): {decision_b.sub_queries}")
    print(f"  -> Reasoning: {decision_b.reasoning}")
    assert decision_b.t0_stable is True
    assert decision_b.should_retrieve is True
    assert decision_b.is_refine is False
    assert len(decision_b.sub_queries) == 1
    print("TEST B PASSED [Single-intent routing]\n")

    # -----------------------------------------------------------------
    # TEST C: Multi-Intent Query -> RETRIEVE with Multiple Subqueries
    # -----------------------------------------------------------------
    print("-------------------------------------------------------")
    print("TEST C: Multi-Intent Query -> RETRIEVE with Multiple Subqueries")
    print("-------------------------------------------------------")
    multi_intent_query = "What are the hotel nightly rate limits and what is the cancellation policy for events?"
    decision_c = await controller.process_incoming_transcript(
        current_transcript=multi_intent_query,
        previous_transcript="",
        session_state=empty_state
    )
    print(f"Query: \"{multi_intent_query}\"")
    print(f"  -> T0 Stable: {decision_c.t0_stable}")
    print(f"  -> Retrieve: {decision_c.should_retrieve}")
    print(f"  -> Is Refine: {decision_c.is_refine}")
    print(f"  -> Sub-Queries ({len(decision_c.sub_queries)}): {decision_c.sub_queries}")
    print(f"  -> Reasoning: {decision_c.reasoning}")
    assert decision_c.t0_stable is True
    assert decision_c.should_retrieve is True
    assert decision_c.is_refine is False
    assert len(decision_c.sub_queries) >= 2
    print("TEST C PASSED [Multi-intent decomposition]\n")

    # -----------------------------------------------------------------
    # TEST D: Follow-up Refinement -> is_refine=True
    # -----------------------------------------------------------------
    print("-------------------------------------------------------")
    print("TEST D: Follow-up Refinement -> is_refine=True")
    print("-------------------------------------------------------")
    # Simulate prior state where travel policy was answered
    active_state = SessionState(
        session_id="test_session_d",
        answer_version=1,
        covered_intents=["general domestic travel per diem"],
        previous_answer="Domestic per diem is $85 per day with standard economy flights.",
        claims=[
            StructuredClaim(
                id=1,
                text="Standard domestic per diem is $85 per day.",
                cites=["travel_policy_04"],
                status=ClaimStatus.SUPPORTED
            )
        ]
    )
    refine_query = "Actually, this is for an international trip to Tokyo."
    decision_d = await controller.process_incoming_transcript(
        current_transcript=refine_query,
        previous_transcript="",
        session_state=active_state
    )
    print(f"Prior Answer: \"{active_state.previous_answer}\"")
    print(f"Refinement Query: \"{refine_query}\"")
    print(f"  -> T0 Stable: {decision_d.t0_stable}")
    print(f"  -> Retrieve: {decision_d.should_retrieve}")
    print(f"  -> Is Refine: {decision_d.is_refine}")
    print(f"  -> Delta Sub-Queries: {decision_d.sub_queries}")
    print(f"  -> Reasoning: {decision_d.reasoning}")
    assert decision_d.t0_stable is True
    assert decision_d.should_retrieve is True
    assert decision_d.is_refine is True
    print("TEST D PASSED [Session refinement / delta query generation]\n")

    # -----------------------------------------------------------------
    # TEST E: Conversational Statement -> retrieve=False (Bypass)
    # -----------------------------------------------------------------
    print("-------------------------------------------------------")
    print("TEST E: Conversational Statement -> retrieve=False (Bypass)")
    print("-------------------------------------------------------")
    conversational_inputs = [
        "Hello!",
        "Thank you, that answers my question.",
        "Okay sounds good",
    ]
    for chat in conversational_inputs:
        decision_e = await controller.process_incoming_transcript(
            current_transcript=chat,
            previous_transcript="",
            session_state=active_state
        )
        print(f"Input: \"{chat}\"")
        print(f"  -> T0 Stable: {decision_e.t0_stable} | Retrieve: {decision_e.should_retrieve} | Sub-Queries: {decision_e.sub_queries}")
        print(f"  -> Reasoning: {decision_e.reasoning}")
        assert decision_e.t0_stable is True
        assert decision_e.should_retrieve is False
        assert len(decision_e.sub_queries) == 0
    print("TEST E PASSED [Conversational bypass without corpus search]\n")

    print("=======================================================")
    print("ALL 5 CONTROLLER TESTS PASSED SUCCESSFULLY!")
    print("=======================================================\n")


@pytest.mark.asyncio
async def test_controller_suite():
    controller = TwoTierController(app_settings=settings)
    empty_state = SessionState(session_id="test_suite")

    # A: Incomplete
    res_a = await controller.process_incoming_transcript("What is the...", "", empty_state)
    assert res_a.t0_stable is False
    assert res_a.should_retrieve is False

    # B: Single intent
    res_b = await controller.process_incoming_transcript("What is the travel policy?", "", empty_state)
    assert res_b.t0_stable is True
    assert res_b.should_retrieve is True
    assert res_b.is_refine is False
    assert len(res_b.sub_queries) == 1

    # C: Multi-intent
    res_c = await controller.process_incoming_transcript("What is the hotel rate and what is the catering limit?", "", empty_state)
    assert res_c.t0_stable is True
    assert res_c.should_retrieve is True
    assert len(res_c.sub_queries) >= 2

    # D: Refine
    refine_state = SessionState(session_id="ref", previous_answer="Domestic rule.")
    res_d = await controller.process_incoming_transcript("Actually, this is for international travel.", "", refine_state)
    assert res_d.is_refine is True
    assert res_d.should_retrieve is True

    # E: Conversational Standalone
    res_e = await controller.process_incoming_transcript("Thank you!", "", empty_state)
    assert res_e.should_retrieve is False
    assert len(res_e.sub_queries) == 0

    # Regression Test 1: Standalone acknowledgements must remain conversational
    standalone_acks = ["Okay", "Thanks", "Thank you.", "Sounds good"]
    for ack in standalone_acks:
        res_ack = await controller.process_incoming_transcript(ack, "", empty_state)
        assert res_ack.t0_stable is True, f"Expected T0 stable for standalone ack: {ack}"
        assert res_ack.should_retrieve is False, f"Expected should_retrieve=False for standalone ack: {ack}"
        assert len(res_ack.sub_queries) == 0

    # Regression Test 2: Conversational prefixes must NOT swallow actual queries
    prefixed_queries = [
        "Okay, what is the hotel policy?",
        "Great, can you check the cancellation policy?",
    ]
    for p_query in prefixed_queries:
        res_p = await controller.process_incoming_transcript(p_query, "", empty_state)
        assert res_p.t0_stable is True, f"Expected T0 stable for query: {p_query}"
        assert res_p.should_retrieve is True, f"Expected should_retrieve=True for query: {p_query}"
        assert len(res_p.sub_queries) >= 1, f"Expected sub_queries for query: {p_query}"


if __name__ == "__main__":
    asyncio.run(run_controller_tests())
