import os
import sys
import asyncio
from pathlib import Path

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
from app.retriever import HybridRetriever
from app.session_state import SessionState, StructuredClaim, ClaimStatus
from app.synthesis import StructuredClaimSynthesizer
from app.verifier import ClaimVerifier
from app.config import settings


async def run_synthesis_grounding_tests():
    print("\n=======================================================")
    print("SESSION + SYNTHESIS + GROUNDING VERIFIER TEST SUITE")
    print("=======================================================\n")

    retriever = HybridRetriever(app_settings=settings)
    retriever.load_indexes()

    synthesizer = StructuredClaimSynthesizer(app_settings=settings)
    verifier = ClaimVerifier(app_settings=settings)

    # -----------------------------------------------------------------
    # TEST 1: Normal Grounded Answer
    # -----------------------------------------------------------------
    print("-------------------------------------------------------")
    print("TEST 1: Normal Grounded Answer")
    print("-------------------------------------------------------")
    session_1 = SessionState(session_id="test_session_1")
    query_1 = "What is the domestic daily meal per diem and receipt policy?"
    
    # Retrieve evidence
    evidence_1 = await retriever.retrieve_for_sub_queries([query_1], session_id=session_1.session_id)
    session_1.add_evidence(evidence_1)
    
    # Synthesize claims
    claims_1 = await synthesizer.generate_structured_claims(query_1, evidence_1, session_1)
    
    # Verify claims
    verified_claims_1, uncertainty_1 = await verifier.verify_claims_pipeline(claims_1, session_1)
    session_1.update_claims(verified_claims_1)
    markdown_1 = synthesizer.render_claims_to_markdown(verified_claims_1, uncertainty_1, session_1.answer_version)
    
    print(f"Query: \"{query_1}\"")
    print(f"Evidence chunks retrieved: {len(evidence_1)}")
    print(f"Claims generated: {len(claims_1)}")
    print(f"Verified claims: {len(verified_claims_1)}")
    print("\nRendered Output:\n" + markdown_1)
    
    assert len(verified_claims_1) > 0
    assert any("travel_policy" in c for claim in verified_claims_1 for c in claim.cites)
    print("\nTEST 1 PASSED [Normal grounded answer]\n")

    # -----------------------------------------------------------------
    # TEST 2: Answer Requiring Multiple Documents
    # -----------------------------------------------------------------
    print("-------------------------------------------------------")
    print("TEST 2: Answer Requiring Multiple Documents")
    print("-------------------------------------------------------")
    session_2 = SessionState(session_id="test_session_2")
    sub_queries_2 = [
        "What are the hotel room rate caps and booking channels?",
        "What is the event venue cancellation policy and penalty schedule?"
    ]
    
    evidence_2 = await retriever.retrieve_for_sub_queries(sub_queries_2, session_id=session_2.session_id)
    session_2.add_evidence(evidence_2)
    
    claims_2 = await synthesizer.generate_structured_claims(
        "Hotel rate caps and event cancellation penalties",
        evidence_2,
        session_2
    )
    verified_claims_2, uncertainty_2 = await verifier.verify_claims_pipeline(claims_2, session_2)
    session_2.update_claims(verified_claims_2)
    markdown_2 = synthesizer.render_claims_to_markdown(verified_claims_2, uncertainty_2, session_2.answer_version)
    
    docs_cited = {ev.doc_id for ev in evidence_2}
    print(f"Sub-queries: {sub_queries_2}")
    print(f"Documents contributing evidence: {docs_cited}")
    print(f"Verified claims count: {len(verified_claims_2)}")
    print("\nRendered Output:\n" + markdown_2)
    
    assert "hotel_policy" in docs_cited
    assert "cancellation_policy" in docs_cited
    print("\nTEST 2 PASSED [Multi-document grounded synthesis]\n")

    # -----------------------------------------------------------------
    # TEST 3: Unsupported Question (Grounded Refusal / Uncertainty)
    # -----------------------------------------------------------------
    print("-------------------------------------------------------")
    print("TEST 3: Unsupported Question (Grounded Refusal / Uncertainty)")
    print("-------------------------------------------------------")
    session_3 = SessionState(session_id="test_session_3")
    query_3 = "What is the cryptocurrency reimbursement and pet travel policy?"
    
    # Simulate empty or irrelevant retrieval for non-existent policy
    evidence_3 = []
    claims_3 = await synthesizer.generate_structured_claims(query_3, evidence_3, session_3)
    verified_claims_3, uncertainty_3 = await verifier.verify_claims_pipeline(claims_3, session_3)
    markdown_3 = synthesizer.render_claims_to_markdown(verified_claims_3, uncertainty_3, session_3.answer_version)
    
    print(f"Unsupported Query: \"{query_3}\"")
    print(f"Claims generated: {len(claims_3)}")
    print(f"Verified claims: {len(verified_claims_3)}")
    print(f"Uncertainty notes: {uncertainty_3}")
    print("\nRendered Output:\n" + markdown_3)
    
    assert len(uncertainty_3) > 0 or any(c.status == ClaimStatus.UNSUPPORTED for c in claims_3)
    print("\nTEST 3 PASSED [Unsupported question handled with uncertainty/refusal]\n")

    # -----------------------------------------------------------------
    # TEST 4, 5, 6: Follow-up Refinement, Citation Preservation, and answer_version Increment
    # -----------------------------------------------------------------
    print("-------------------------------------------------------")
    print("TEST 4, 5, 6: Refinement + Citation Preservation + answer_version Increment")
    print("-------------------------------------------------------")
    # Turn 1: Initial query
    session_4 = SessionState(session_id="test_session_refine")
    turn_1_query = "What is the expense reimbursement submission timeline?"
    session_4.record_refinement(turn_1_query)
    assert session_4.answer_version == 1
    
    ev_turn1 = await retriever.retrieve_for_sub_queries([turn_1_query], session_id=session_4.session_id)
    session_4.add_evidence(ev_turn1)
    claims_turn1 = await synthesizer.generate_structured_claims(turn_1_query, ev_turn1, session_4)
    v_claims_1, unc_1 = await verifier.verify_claims_pipeline(claims_turn1, session_4)
    session_4.update_claims(v_claims_1)
    session_4.previous_answer = synthesizer.render_claims_to_markdown(v_claims_1, unc_1, session_4.answer_version)
    
    print(f"Turn 1 Query: \"{turn_1_query}\" (Version: v{session_4.answer_version})")
    print(f"Evidence pool size: {len(session_4.evidence_pool)}")
    print(f"Turn 1 Claims: {len(v_claims_1)}")
    
    # Turn 2: Delta refinement
    turn_2_refine = "Actually, I mean international travel currency conversion rules."
    session_4.record_refinement(turn_2_refine)
    assert session_4.answer_version == 2
    
    # Delta retrieval
    ev_turn2 = await retriever.retrieve_for_sub_queries([turn_2_refine], session_id=session_4.session_id)
    session_4.add_evidence(ev_turn2)
    
    # Synthesize with refinement=True
    claims_turn2 = await synthesizer.generate_structured_claims(
        user_query=turn_2_refine,
        evidence=ev_turn2,
        session_state=session_4,
        is_refinement=True
    )
    v_claims_2, unc_2 = await verifier.verify_claims_pipeline(claims_turn2, session_4)
    session_4.update_claims(v_claims_2)
    session_4.previous_answer = synthesizer.render_claims_to_markdown(v_claims_2, unc_2, session_4.answer_version)
    
    print(f"\nTurn 2 Refinement: \"{turn_2_refine}\" (Version: v{session_4.answer_version})")
    print(f"Updated Evidence pool size: {len(session_4.evidence_pool)}")
    print(f"Updated Total Claims: {len(v_claims_2)}")
    print(f"Total Unique Citations: {session_4.citations}")
    print("\nTurn 2 Rendered Output:\n" + session_4.previous_answer)
    
    # Assertions
    assert session_4.answer_version == 2, "answer_version should be incremented to 2"
    assert len(session_4.citations) >= 2, "Citations from both turns should be preserved in session state"
    assert any("[DOC: " in cite for cite in session_4.citations), "Citation format must match [DOC: ...]"
    
    print("\nTESTS 4, 5, 6 PASSED [Refinement, Citations, Version Increment]\n")
    print("=======================================================")
    print("ALL 6 TESTS PASSED SUCCESSFULLY!")
    print("=======================================================\n")


@pytest.mark.asyncio
async def test_session_synthesis_grounding():
    retriever = HybridRetriever(app_settings=settings)
    retriever.load_indexes()
    synthesizer = StructuredClaimSynthesizer(app_settings=settings)
    verifier = ClaimVerifier(app_settings=settings)

    session = SessionState(session_id="pytest_session")
    session.record_refinement("Initial query")
    assert session.answer_version == 1

    ev = await retriever.retrieve_for_sub_queries(["hotel nightly rate limits"])
    session.add_evidence(ev)
    assert len(session.evidence_pool) > 0

    claims = await synthesizer.generate_structured_claims("hotel limits", ev, session)
    assert len(claims) > 0

    v_claims, unc = await verifier.verify_claims_pipeline(claims, session)
    assert len(v_claims) > 0
    session.update_claims(v_claims)
    assert len(session.citations) > 0


if __name__ == "__main__":
    asyncio.run(run_synthesis_grounding_tests())
