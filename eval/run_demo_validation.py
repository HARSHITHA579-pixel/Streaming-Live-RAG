"""
End-to-End Live Hackathon Demo Validation Script

Runs the 5 demo turns one-by-one through the real PipelineOrchestrator without modifying any code.
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from app.config import settings
from app.main import PipelineOrchestrator, active_sessions


async def run_turn(orchestrator: PipelineOrchestrator, session_id: str, query_text: str) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []

    async def emit_callback(event_payload: Dict[str, Any]):
        events.append(event_payload)

    await orchestrator.process_streaming_transcript(
        session_id=session_id,
        transcript_chunk=query_text,
        event_callback=emit_callback,
        is_final=True
    )

    session_state = active_sessions.get(session_id)

    # Categorize events
    ctrl_events = [e for e in events if e.get("event") == "controller"]
    ret_events = [e for e in events if e.get("event") == "retrieval"]
    claim_events = [e for e in events if e.get("event") == "claim"]
    ver_events = [e for e in events if e.get("event") == "verification"]
    ans_events = [e for e in events if e.get("event") == "answer"]

    return {
        "session_id": session_id,
        "query": query_text,
        "events": events,
        "ctrl_events": ctrl_events,
        "ret_events": ret_events,
        "claim_events": claim_events,
        "ver_events": ver_events,
        "ans_events": ans_events,
        "session_state": session_state
    }


async def main():
    orchestrator = PipelineOrchestrator()
    results = []

    print("\n" + "=" * 75)
    print("STARTING FOCUSED END-TO-END DEMO VALIDATION")
    print("=" * 75)

    # -------------------------------------------------------------
    # DEMO 1 — GENERAL TRAVEL REIMBURSEMENT (Turn 1)
    # -------------------------------------------------------------
    print("\n>>> Running Demo 1 Turn 1...")
    d1_t1 = await run_turn(orchestrator, "demo1_travel", "Summarize the travel reimbursement rule for an employee trip.")
    results.append(("DEMO 1 — GENERAL TRAVEL REIMBURSEMENT", d1_t1))

    # -------------------------------------------------------------
    # DEMO 1 — REFINEMENT (Turn 2, Same Session)
    # -------------------------------------------------------------
    print(">>> Running Demo 1 Turn 2 (Refinement in same session)...")
    d1_t2 = await run_turn(orchestrator, "demo1_travel", "The trip was international and the booking was made after travel.")
    results.append(("DEMO 1 — REFINEMENT", d1_t2))

    # -------------------------------------------------------------
    # DEMO 2 — PUNE VENUE (Turn 1)
    # -------------------------------------------------------------
    print(">>> Running Demo 2 Turn 1 (Pune venue)...")
    d2_t1 = await run_turn(orchestrator, "demo2_pune", "I need to plan a customer workshop in Pune for 30 people.")
    results.append(("DEMO 2 — PUNE VENUE", d2_t1))

    # -------------------------------------------------------------
    # DEMO 2 — ADDITIONAL REQUIREMENTS (Turn 2, Same Session)
    # -------------------------------------------------------------
    print(">>> Running Demo 2 Turn 2 (Additive refinement in same session)...")
    d2_t2 = await run_turn(orchestrator, "demo2_pune", "I need the cancellation policy and the catering options.")
    results.append(("DEMO 2 — ADDITIONAL REQUIREMENTS", d2_t2))

    # -------------------------------------------------------------
    # OUT-OF-CORPUS SAFETY CHECK
    # -------------------------------------------------------------
    print(">>> Running Out-of-Corpus Safety Check...")
    d3_safe = await run_turn(orchestrator, "demo3_safety_check", "What is the fire safety procedure for a large crowd?")
    results.append(("OUT-OF-CORPUS SAFETY CHECK", d3_safe))

    print("\n" + "=" * 75)
    print("VALIDATION EXECUTION COMPLETE. GENERATING DETAILED REPORT...")
    print("=" * 75 + "\n")

    for title, res in results:
        print("=" * 75)
        print(f"REPORT: {title}")
        print("=" * 75)
        print(f"Query:               \"{res['query']}\"")
        print(f"Session ID:          {res['session_id']}")

        # T0 & T1
        if res['ctrl_events']:
            ctrl = res['ctrl_events'][-1]
            print(f"T0 Stability:        {'Stable (PROCEED)' if ctrl.get('t0_stable') else 'WAIT'}")
            print(f"T1 Intent:           Sub-queries: {ctrl.get('sub_queries')} | should_retrieve: {ctrl.get('should_retrieve')} | is_refine: {ctrl.get('is_refine')} | refinement_type: {ctrl.get('refinement_type')}")
        else:
            print("T0 / T1:             No controller event emitted")

        # Retrieval & Evidence Gate
        if res['ret_events']:
            ret = res['ret_events'][-1]
            gate_status = ret.get('gate_status', 'supported')
            print(f"Retrieval Status:    Triggered ({ret.get('evidence_count', 0)} chunks retrieved)")
            print(f"Evidence Gate:       Status: {gate_status} | Reason: {ret.get('gate_reason', 'N/A')}")
            chunks = ret.get('chunks', [])
            print("Top Sources Retrieved:")
            for i, c in enumerate(chunks[:5], start=1):
                print(f"  [{i}] Doc: {c.get('doc_id')} | Category: {c.get('category')} | Scope: {c.get('geographic_scope')} | Section: {c.get('section_title')} (Chunk: {c.get('chunk_id')})")
        else:
            print("Retrieval Status:    Bypassed / None")
            print("Evidence Gate:       Bypassed")

        # Synthesis & Grounding
        if res['ver_events']:
            print(f"Synthesis Status:    Generated {len(res['claim_events'])} raw claims")
            print(f"Grounding Status:    {len(res['ver_events'])} claims verified")
            for i, v in enumerate(res['ver_events'], start=1):
                status = v.get('status', 'unknown')
                cites = v.get('cites', [])
                print(f"  Claim {i} [{status.upper()}]: \"{v.get('text')}\" (Citations: {cites})")
        else:
            print("Synthesis Status:    Skipped / Not triggered")
            print("Grounding Status:    No claims generated/verified")

        # Final Answer
        if res['ans_events']:
            ans = res['ans_events'][-1]
            print(f"Answer Version:      v{ans.get('answer_version', 1)}")
            print(f"Citations Count:     {len(ans.get('citations', []))}")
            print(f"Total Latency:       {ans.get('total_latency_ms', 0):.2f} ms")
            print("Final Synthesized Answer:")
            print("-" * 50)
            print(ans.get('answer', ''))
            print("-" * 50)
        else:
            print("Final Answer:        None emitted")

        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
